pub mod backend;
pub mod accelerated;
pub mod transformer;

use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh64::Xxh64;

pub use transformer::{generate_token_ids, run_model, run_model_with_options};

pub const MAGIC: [u8; 4] = *b"AXON";
pub const VERSION_MAJOR: u16 = 2;
pub const VERSION_MINOR: u16 = 0;
pub const FLAG_HAS_CODEBOOKS: u32 = 1 << 0;
pub const FLAG_HAS_OUTLIER_SPINE: u32 = 1 << 1;
pub const FLAG_STREAM_ORDERED: u32 = 1 << 2;
pub const FLAG_HAS_HW_HINTS: u32 = 1 << 3;
pub const FLAG_HEADER_ZSTD: u32 = 1 << 4;
pub const FLAG_SCALES_INTERLEAVED: u32 = 1 << 5;
pub const FLAG_HAS_CHECKSUMS: u32 = 1 << 6;
pub const FLAG_MXQ_V2: u32 = 1 << 7;
pub const FLAG_HAS_BOOT_REGION: u32 = 1 << 8;
pub const FLAG_HAS_KV_HINTS: u32 = 1 << 9;
pub const FLAG_HAS_DEP_GRAPH: u32 = 1 << 10;
pub const FLAG_HAS_SPECULATIVE_DRAFT: u32 = 1 << 11;
pub const FLAG_HAS_EXPERT_DEDUP: u32 = 1 << 12;
pub const FLAG_HAS_LORA_DELTA: u32 = 1 << 13;
pub const FLAG_PER_HEAD_QUANT: u32 = 1 << 14;
pub const FLAG_NF_QUANT: u32 = 1 << 15;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub manifest_version: String,
    pub task: String,
    pub model_file: String,
    #[serde(default)]
    pub config_file: Option<String>,
    #[serde(default)]
    pub generation_config_file: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<TokenizerManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerManifest {
    pub kind: String,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_dim: u64,
    pub intermediate_dim: u64,
    pub num_layers: u64,
    pub num_attention_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub context_length: u64,
    pub vocab_size: u64,
    pub rope: RopeConfig,
    #[serde(default)]
    pub total_parameter_count: Option<u64>,
    #[serde(default)]
    pub active_parameter_count: Option<u64>,
    #[serde(default)]
    pub moe: Option<MoeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub num_experts: u64,
    pub experts_per_token: u64,
    pub expert_intermediate_dim: u64,
    #[serde(default)]
    pub num_shared_experts: Option<u64>,
    #[serde(default)]
    pub shared_expert_intermediate_dim: Option<u64>,
    #[serde(default)]
    pub router_aux_loss_coef: Option<f64>,
    #[serde(default)]
    pub expert_layer_frequency: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    #[serde(rename = "type")]
    pub rope_type: String,
    #[serde(default)]
    pub theta: Option<f64>,
    #[serde(default)]
    pub scaling: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    pub default_generation: GenerationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u64,
    pub max_new_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub format: String,
    pub identifier: String,
    pub conversion_tool: String,
    pub conversion_time_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    pub dataset: String,
    pub tokens: u64,
    pub perplexity_fp16: f64,
    pub perplexity_axon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactMoeInfo {
    pub num_experts: u64,
    pub active_experts: u64,
    pub expert_hidden_dim: u64,
    #[serde(default)]
    pub expert_similarity_dedup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheLayerHint {
    pub layer: i64,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheHints {
    pub default_dtype: String,
    #[serde(default)]
    pub per_layer: Vec<KvCacheLayerHint>,
    pub max_seq_kv_budget_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDependencyGraph {
    #[serde(default)]
    pub parallel_groups: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDraftInfo {
    pub arch: String,
    pub hidden_dim: u64,
    pub num_layers: u64,
    pub vocab_size: u64,
    pub draft_bytes: u64,
    pub draft_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothQuantScaleDescriptor {
    pub offset: u64,
    pub channels: u64,
    pub dtype: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDedupInfo {
    pub region_offset: u64,
    pub region_bytes: u64,
    #[serde(default)]
    pub canonical_map: BTreeMap<String, String>,
    pub corrections_offset: u64,
    pub similarity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraInfo {
    pub base_model: String,
    pub base_hash: String,
    pub rank: u64,
    pub alpha: u64,
    #[serde(default)]
    pub target_modules: Vec<String>,
    pub region_offset: u64,
    pub region_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaHints {
    pub num_nodes: u64,
    #[serde(default)]
    pub tensor_node_map: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub shape: Vec<u64>,
    pub dtype: String,
    #[serde(default)]
    pub bits: Option<u8>,
    #[serde(default)]
    pub group_size: Option<u64>,
    #[serde(default)]
    pub source_tensor_name: Option<String>,
    #[serde(default)]
    pub data_offset: u64,
    #[serde(default)]
    pub data_bytes: u64,
    #[serde(default)]
    pub scale_interleaved: bool,
    #[serde(default)]
    pub outlier_indices_offset: Option<u64>,
    #[serde(default)]
    pub outlier_count: Option<u64>,
    #[serde(default)]
    pub sensitivity_score: Option<f64>,
    pub stream_order: u32,
    #[serde(default)]
    pub per_head_bits: Option<Vec<u8>>,
    #[serde(default)]
    pub nf_scale_fp16: bool,
    #[serde(default)]
    pub smoothquant_scale: Option<String>,
    #[serde(default)]
    pub prefetch_priority: Option<f64>,
    #[serde(default)]
    pub codebook_id: Option<String>,
    #[serde(default)]
    pub vq_dim: Option<u64>,
    #[serde(default)]
    pub dedup_canonical: Option<String>,
    #[serde(default)]
    pub dedup_correction_offset: Option<u64>,
    #[serde(default)]
    pub dedup_correction_count: Option<u64>,
    #[serde(default)]
    pub lora_rank: Option<u64>,
    #[serde(default)]
    pub lora_alpha: Option<u64>,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub checksum_xxh64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookDescriptor {
    pub offset: u64,
    pub entries: u64,
    pub dim: u64,
    pub dtype: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareHint {
    pub kernel: String,
    #[serde(default)]
    pub tile: Option<Vec<u64>>,
    #[serde(default)]
    pub unroll: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub format: String,
    pub version: String,
    pub task: String,
    pub architecture: String,
    pub model_family: String,
    pub model: ModelConfig,
    pub runtime: RuntimeConfig,
    pub source: SourceInfo,
    #[serde(default)]
    pub total_params: Option<u64>,
    #[serde(default)]
    pub active_params: Option<u64>,
    #[serde(default)]
    pub moe: Option<CompactMoeInfo>,
    #[serde(default)]
    pub boot_region_bytes: Option<u64>,
    #[serde(default)]
    pub kv_cache_hints: Option<KvCacheHints>,
    #[serde(default)]
    pub tensor_dep_graph: Option<TensorDependencyGraph>,
    #[serde(default)]
    pub speculative_draft: Option<SpeculativeDraftInfo>,
    #[serde(default)]
    pub smoothquant_scales: BTreeMap<String, SmoothQuantScaleDescriptor>,
    #[serde(default)]
    pub expert_dedup: Option<ExpertDedupInfo>,
    #[serde(default)]
    pub lora: Option<LoraInfo>,
    #[serde(default)]
    pub numa_hints: Option<NumaHints>,
    pub avg_bits_per_weight: f64,
    pub quant_method: String,
    #[serde(default)]
    pub calibration: Option<CalibrationInfo>,
    pub tensors: BTreeMap<String, TensorDescriptor>,
    #[serde(default)]
    pub codebooks: BTreeMap<String, CodebookDescriptor>,
    #[serde(default)]
    pub hw_hints: BTreeMap<String, HardwareHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleHeader {
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: u32,
    pub header_len: u32,
    pub data_offset: u64,
    pub outlier_offset: u64,
    pub codebook_offset: u64,
    pub tail_offset: u64,
    pub boot_cutoff: u8,
    pub speculative_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleSummary {
    pub manifest: BundleManifest,
    pub metadata: Metadata,
    pub file_size: u64,
    pub flags: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTag {
    pub name: String,
    pub path: String,
    pub architecture: String,
    pub model_family: String,
    pub tensor_count: usize,
    pub file_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProcessInfo {
    pub name: String,
    pub path: String,
    pub architecture: String,
    pub model_family: String,
    pub backend: String,
    pub device: String,
    pub file_size: u64,
    pub loaded_at_unix: u64,
    pub last_used_unix: u64,
    pub request_count: u64,
    pub active_requests: usize,
    pub state: String,
}

#[derive(Debug, Clone)]
pub struct RegisteredModel {
    pub name: String,
    pub path: PathBuf,
    pub summary: BundleSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleStats {
    pub quantized_tensors: usize,
    pub vq_tensors: usize,
    pub outlier_tensors: usize,
    pub codebook_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPreview {
    pub model: String,
    pub prompt: String,
    pub response: String,
    pub done: bool,
    pub done_reason: String,
    pub message: String,
    pub backend: backend::BackendSelection,
    pub stats: BundleStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunOptions {
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub raw_prompt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowResponse {
    pub summary: BundleSummary,
    pub backend: backend::BackendSelection,
    pub stats: BundleStats,
}

pub fn load_bundle(bundle_dir: &Path) -> Result<BundleSummary> {
    let manifest_path = bundle_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: BundleManifest = serde_json::from_str(&manifest_text).context("failed to parse manifest.json")?;
    validate_manifest_files(bundle_dir, &manifest)?;

    let model_path = bundle_dir.join(&manifest.model_file);
    let mut file = File::open(&model_path).with_context(|| format!("failed to open {}", model_path.display()))?;
    let file_size = file.metadata()?.len();
    let header = read_header(&mut file)?;
    validate_header(&header)?;
    let metadata_bytes = read_metadata_bytes(&mut file, &header)?;
    let metadata: Metadata = serde_json::from_slice(&metadata_bytes).context("failed to parse metadata JSON")?;
    validate_metadata(&metadata, &header)?;
    validate_tensor_regions(&mut file, &header, &metadata, file_size)?;
    validate_outlier_regions(&header, &metadata, file_size)?;
    validate_codebook_regions(&header, &metadata, file_size)?;
    validate_expert_dedup_regions(&metadata, file_size)?;
    validate_speculative_region(&header, &metadata, file_size)?;

    Ok(BundleSummary {
        manifest,
        metadata,
        file_size,
        flags: header.flags,
    })
}

pub fn scan_registry(root: &Path) -> Result<Vec<RegisteredModel>> {
    let mut models = Vec::new();
    if !root.exists() {
        return Ok(models);
    }
    if is_bundle_dir(root) {
        let summary = load_bundle(root)?;
        let name = root
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        models.push(RegisteredModel {
            name,
            path: root.to_path_buf(),
            summary,
        });
        return Ok(models);
    }

    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || !is_bundle_dir(&path) {
            continue;
        }
        let name = path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        let summary = load_bundle(&path)?;
        models.push(RegisteredModel { name, path, summary });
    }
    models.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(models)
}

pub fn tags_from_registry(models: &[RegisteredModel]) -> Vec<ModelTag> {
    models
        .iter()
        .map(|model| ModelTag {
            name: model.name.clone(),
            path: model.path.display().to_string(),
            architecture: model.summary.metadata.architecture.clone(),
            model_family: model.summary.metadata.model_family.clone(),
            tensor_count: model.summary.metadata.tensors.len(),
            file_size: model.summary.file_size,
        })
        .collect()
}

pub fn get_registered_model<'a>(models: &'a [RegisteredModel], name: &str) -> Option<&'a RegisteredModel> {
    models.iter().find(|model| model.name == name)
}

pub fn bundle_stats(summary: &BundleSummary) -> BundleStats {
    let mut quantized = 0;
    let mut vq = 0;
    let mut outliers = 0;
    for tensor in summary.metadata.tensors.values() {
        if matches!(tensor.dtype.as_str(), "axon_mxq" | "axon_nf2" | "axon_nf3") {
            quantized += 1;
        }
        if tensor.dtype == "axon_vq" {
            vq += 1;
        }
        if tensor.outlier_count.unwrap_or(0) > 0 {
            outliers += 1;
        }
    }
    BundleStats {
        quantized_tensors: quantized,
        vq_tensors: vq,
        outlier_tensors: outliers,
        codebook_count: summary.metadata.codebooks.len(),
    }
}

pub fn show_bundle(bundle_dir: &Path, preferred_backend: Option<&str>) -> Result<ShowResponse> {
    let summary = load_runtime_summary(bundle_dir)?;
    let backend = backend::select_backend(&summary, preferred_backend);
    Ok(ShowResponse {
        stats: bundle_stats(&summary),
        summary,
        backend,
    })
}

pub fn run_preview(bundle_dir: &Path, prompt: &str, preferred_backend: Option<&str>) -> Result<GenerationPreview> {
    transformer::run_model(bundle_dir, prompt, preferred_backend, None)
}

pub fn decode_named_tensor(bundle_dir: &Path, tensor_name: &str) -> Result<Vec<f32>> {
    let summary = load_bundle(bundle_dir)?;
    let (header, mut file) = open_model_file(bundle_dir, &summary.manifest)?;
    let tensor = summary
        .metadata
        .tensors
        .get(tensor_name)
        .ok_or_else(|| anyhow!("unknown tensor {tensor_name}"))?;
    let decode_tensor = if let Some(canonical) = tensor.dedup_canonical.as_deref() {
        summary
            .metadata
            .tensors
            .get(canonical)
            .ok_or_else(|| anyhow!("missing canonical tensor {canonical} for dedup tensor"))?
    } else {
        tensor
    };
    let data = read_region_bytes(
        &mut file,
        tensor_absolute_start(&header, &summary.metadata, decode_tensor)?,
        decode_tensor.data_bytes,
    )?;
    let mut decoded = match decode_tensor.dtype.as_str() {
        "fp16" => decode_f16_bytes(&data),
        "bf16" => decode_bf16_bytes(&data),
        "fp32" => decode_f32_bytes(&data),
        "axon_mxq" => decode_mxq_tensor(decode_tensor, &data)?,
        "axon_nf2" | "axon_nf3" => decode_nf_tensor(decode_tensor, &data)?,
        "axon_vq" => decode_vq_tensor(&mut file, &header, &summary.metadata, decode_tensor, &data)?,
        other => bail!("unsupported tensor dtype {other}"),
    };
    if decode_tensor.outlier_count.unwrap_or(0) > 0 {
        apply_outliers(&mut file, &header, decode_tensor, &mut decoded)?;
    }
    if tensor.dedup_canonical.is_some() {
        apply_dedup_corrections(&mut file, &summary.metadata, tensor, &mut decoded)?;
    }
    Ok(decoded)
}

fn open_model_file(bundle_dir: &Path, manifest: &BundleManifest) -> Result<(BundleHeader, File)> {
    let model_path = bundle_dir.join(&manifest.model_file);
    let mut file = File::open(&model_path).with_context(|| format!("failed to open {}", model_path.display()))?;
    let header = read_header(&mut file)?;
    Ok((header, file))
}

fn is_bundle_dir(path: &Path) -> bool {
    path.join("manifest.json").is_file()
}

pub(crate) fn default_models_dir() -> PathBuf {
    if let Ok(configured) = env::var("AXONAL_MODELS") {
        if !configured.trim().is_empty() {
            return PathBuf::from(configured);
        }
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".axonal").join("models");
    }
    PathBuf::from("./models")
}

fn normalize_bundle_candidate(candidate: &Path) -> Option<PathBuf> {
    if candidate.is_dir() && is_bundle_dir(candidate) {
        return Some(candidate.to_path_buf());
    }
    if candidate.is_file() {
        if let Some(parent) = candidate.parent() {
            if is_bundle_dir(parent) {
                return Some(parent.to_path_buf());
            }
        }
    }
    None
}

pub(crate) fn resolve_lora_base_bundle(bundle_dir: &Path, lora: &LoraInfo) -> Result<PathBuf> {
    let label = lora.base_model.trim();
    let stem = Path::new(label)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(label);
    let mut candidates = Vec::new();
    let label_path = PathBuf::from(label);
    if label_path.is_absolute() {
        candidates.push(label_path.clone());
    }
    candidates.push(bundle_dir.join(label));
    if let Some(parent) = bundle_dir.parent() {
        candidates.push(parent.join(label));
        candidates.push(parent.join(stem));
    }
    let library = default_models_dir();
    candidates.push(library.join(label));
    candidates.push(library.join(stem));

    for candidate in candidates {
        if let Some(bundle) = normalize_bundle_candidate(&candidate) {
            return Ok(bundle);
        }
    }

    if let Ok(models) = scan_registry(&library) {
        if let Some(model) = models.iter().find(|model| model.name == stem) {
            return Ok(model.path.clone());
        }
    }

    bail!(
        "unable to resolve base bundle for LoRA delta {}; looked for {} near {} and in {}",
        lora.base_model,
        stem,
        bundle_dir.display(),
        library.display()
    );
}

pub(crate) fn load_runtime_summary(bundle_dir: &Path) -> Result<BundleSummary> {
    let raw = load_bundle(bundle_dir)?;
    let Some(lora) = raw.metadata.lora.clone() else {
        return Ok(raw);
    };
    let base_bundle = resolve_lora_base_bundle(bundle_dir, &lora)?;
    let mut base = load_bundle(&base_bundle)?;
    base.manifest = raw.manifest.clone();
    base.metadata.runtime = raw.metadata.runtime.clone();
    base.metadata.source = raw.metadata.source.clone();
    base.metadata.lora = raw.metadata.lora.clone();
    base.file_size = raw.file_size;
    base.flags = raw.flags;
    Ok(base)
}

fn validate_manifest_files(bundle_dir: &Path, manifest: &BundleManifest) -> Result<()> {
    let model_path = bundle_dir.join(&manifest.model_file);
    if !model_path.is_file() {
        bail!("missing model file {}", model_path.display());
    }
    if let Some(config_file) = &manifest.config_file {
        ensure_bundle_file(bundle_dir, config_file)?;
    }
    if let Some(generation_config_file) = &manifest.generation_config_file {
        ensure_bundle_file(bundle_dir, generation_config_file)?;
    }
    if let Some(tokenizer) = &manifest.tokenizer {
        for file in &tokenizer.files {
            ensure_bundle_file(bundle_dir, file)?;
        }
    }
    Ok(())
}

fn ensure_bundle_file(bundle_dir: &Path, file_name: &str) -> Result<()> {
    let path = bundle_dir.join(file_name);
    if !path.is_file() {
        bail!("missing bundle asset {}", path.display());
    }
    Ok(())
}

fn read_header(file: &mut File) -> Result<BundleHeader> {
    let mut bytes = [0_u8; 64];
    file.read_exact(&mut bytes)?;
    if bytes[0..4] != MAGIC {
        bail!("invalid AXON magic");
    }
    if bytes[49..52].iter().any(|byte| *byte != 0) || bytes[60..64].iter().any(|byte| *byte != 0) {
        bail!("reserved AXON header bytes must be zero");
    }
    Ok(BundleHeader {
        version_major: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
        version_minor: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
        flags: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        header_len: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        data_offset: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        outlier_offset: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        codebook_offset: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
        tail_offset: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
        boot_cutoff: bytes[48],
        speculative_offset: u64::from_le_bytes(bytes[52..60].try_into().unwrap()),
    })
}

fn validate_header(header: &BundleHeader) -> Result<()> {
    if header.version_major > VERSION_MAJOR {
        bail!("unsupported AXON version {}.{}", header.version_major, header.version_minor);
    }
    if header.data_offset < 64 {
        bail!("data offset is invalid");
    }
    if header.flags & FLAG_HAS_BOOT_REGION == 0 {
        if header.tail_offset != 0 || header.boot_cutoff != 0 {
            bail!("tail_offset and boot_cutoff must be zero when HAS_BOOT_REGION is not set");
        }
    } else if header.tail_offset < header.data_offset {
        bail!("tail_offset must not precede data_offset");
    }
    if header.flags & FLAG_HAS_SPECULATIVE_DRAFT == 0 && header.speculative_offset != 0 {
        bail!("speculative_offset must be zero when HAS_SPECULATIVE_DRAFT is not set");
    }
    Ok(())
}

fn read_metadata_bytes(file: &mut File, header: &BundleHeader) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(64))?;
    let mut metadata_bytes = vec![0_u8; header.header_len as usize];
    file.read_exact(&mut metadata_bytes)?;
    if header.flags & FLAG_HEADER_ZSTD != 0 {
        return zstd::stream::decode_all(&metadata_bytes[..]).context("failed to decompress metadata");
    }
    Ok(metadata_bytes)
}

fn validate_metadata(metadata: &Metadata, header: &BundleHeader) -> Result<()> {
    if metadata.format != "axon" {
        bail!("metadata format must be axon");
    }
    if metadata.task != "causal_lm" {
        bail!("unsupported task {}", metadata.task);
    }
    if metadata.tensors.is_empty() {
        bail!("metadata contains no tensors");
    }
    if !metadata.version.starts_with(&format!("{}.{}", header.version_major, header.version_minor)) {
        bail!("metadata version {} does not match header", metadata.version);
    }
    validate_model_config(&metadata.model)?;
    if let Some(total) = metadata.total_params {
        if total == 0 {
            bail!("metadata total_params must be greater than zero");
        }
    }
    if let Some(active) = metadata.active_params {
        if active == 0 {
            bail!("metadata active_params must be greater than zero");
        }
        if let Some(total) = metadata.total_params {
            if active > total {
                bail!("metadata active_params exceeds total_params");
            }
        }
    }
    if let Some(moe) = &metadata.moe {
        if moe.num_experts == 0 || moe.active_experts == 0 || moe.expert_hidden_dim == 0 {
            bail!("metadata moe fields must be greater than zero");
        }
        if moe.active_experts > moe.num_experts {
            bail!("metadata moe.active_experts exceeds moe.num_experts");
        }
    }
    if let Some(kv_hints) = &metadata.kv_cache_hints {
        if kv_hints.default_dtype.is_empty() {
            bail!("kv_cache_hints.default_dtype must not be empty");
        }
    }
    if let Some(graph) = &metadata.tensor_dep_graph {
        for group in &graph.parallel_groups {
            if group.is_empty() {
                bail!("tensor_dep_graph.parallel_groups must not contain empty groups");
            }
        }
    }
    if let Some(speculative) = &metadata.speculative_draft {
        if speculative.draft_bytes == 0 {
            bail!("speculative_draft.draft_bytes must be greater than zero");
        }
    }
    for (scale_name, scale) in &metadata.smoothquant_scales {
        if scale.channels == 0 || scale.size == 0 {
            bail!("smoothquant scale {scale_name} has invalid size");
        }
    }
    if let Some(dedup) = &metadata.expert_dedup {
        if dedup.region_bytes == 0 {
            bail!("expert_dedup.region_bytes must be greater than zero");
        }
    }
    if let Some(lora) = &metadata.lora {
        if lora.rank == 0 || lora.alpha == 0 || lora.region_bytes == 0 {
            bail!("lora metadata contains invalid zero values");
        }
    }
    if let Some(numa) = &metadata.numa_hints {
        if numa.num_nodes == 0 {
            bail!("numa_hints.num_nodes must be greater than zero");
        }
    }
    if metadata.boot_region_bytes.unwrap_or(0) > 0 && header.tail_offset == 0 {
        bail!("metadata boot_region_bytes is set but header tail_offset is zero");
    }
    if metadata.speculative_draft.is_some() && header.speculative_offset == 0 {
        bail!("metadata speculative_draft is set but header speculative_offset is zero");
    }
    if let Some(lora) = &metadata.lora {
        if lora.region_offset == 0 {
            bail!("metadata lora.region_offset must be non-zero when lora metadata is present");
        }
    }
    Ok(())
}

fn validate_model_config(model: &ModelConfig) -> Result<()> {
    if let Some(total) = model.total_parameter_count {
        if total == 0 {
            bail!("model total_parameter_count must be greater than zero");
        }
    }
    if let Some(active) = model.active_parameter_count {
        if active == 0 {
            bail!("model active_parameter_count must be greater than zero");
        }
        if let Some(total) = model.total_parameter_count {
            if active > total {
                bail!("model active_parameter_count exceeds total_parameter_count");
            }
        }
    }
    if let Some(moe) = &model.moe {
        if moe.num_experts == 0 {
            bail!("model moe.num_experts must be greater than zero");
        }
        if moe.experts_per_token == 0 {
            bail!("model moe.experts_per_token must be greater than zero");
        }
        if moe.experts_per_token > moe.num_experts {
            bail!("model moe.experts_per_token exceeds moe.num_experts");
        }
        if moe.expert_intermediate_dim == 0 {
            bail!("model moe.expert_intermediate_dim must be greater than zero");
        }
        if let Some(shared) = moe.num_shared_experts {
            if shared == 0 {
                bail!("model moe.num_shared_experts must be greater than zero when present");
            }
        }
        if let Some(shared_dim) = moe.shared_expert_intermediate_dim {
            if shared_dim == 0 {
                bail!("model moe.shared_expert_intermediate_dim must be greater than zero when present");
            }
        }
        if let Some(layer_frequency) = moe.expert_layer_frequency {
            if layer_frequency == 0 {
                bail!("model moe.expert_layer_frequency must be greater than zero when present");
            }
        }
    }
    Ok(())
}

fn validate_tensor_regions(file: &mut File, header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (name, tensor) in &metadata.tensors {
        if let Some(canonical) = &tensor.dedup_canonical {
            if !metadata.tensors.contains_key(canonical) {
                bail!("tensor {name} references missing dedup canonical {canonical}");
            }
            if tensor.dedup_correction_offset.is_none() || tensor.dedup_correction_count.is_none() {
                bail!("tensor {name} is missing dedup correction metadata");
            }
            continue;
        }
        let start = tensor_absolute_start(header, metadata, tensor)?;
        let end = start
            .checked_add(tensor.data_bytes)
            .ok_or_else(|| anyhow!("tensor region overflow for {name}"))?;
        if end > file_size {
            bail!("tensor {name} exceeds file bounds");
        }
        if let Some(codebook_id) = &tensor.codebook_id {
            if !metadata.codebooks.contains_key(codebook_id) {
                bail!("tensor {name} references missing codebook {codebook_id}");
            }
        }
        if let Some(scale_name) = &tensor.smoothquant_scale {
            if !metadata.smoothquant_scales.contains_key(scale_name) {
                bail!("tensor {name} references missing smoothquant scale {scale_name}");
            }
        }
        if let Some(priority) = tensor.prefetch_priority {
            if !(0.0..=1.0).contains(&priority) {
                bail!("tensor {name} has invalid prefetch_priority");
            }
        }
        if header.flags & FLAG_HAS_CHECKSUMS != 0 {
            let expected = tensor
                .checksum_xxh64
                .as_deref()
                .ok_or_else(|| anyhow!("tensor {name} is missing checksum_xxh64"))?;
            let actual = checksum_file_region(file, start, tensor.data_bytes)?;
            if actual != expected {
                bail!("checksum mismatch for tensor {name}");
            }
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    for window in ranges.windows(2) {
        if window[0].1 > window[1].0 {
            bail!("tensor regions overlap: {} and {}", window[0].2, window[1].2);
        }
    }
    Ok(())
}

fn validate_outlier_regions(header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    for (name, tensor) in &metadata.tensors {
        let Some(offset) = tensor.outlier_indices_offset else {
            continue;
        };
        let outlier_count = tensor
            .outlier_count
            .ok_or_else(|| anyhow!("tensor {name} is missing outlier_count"))?;
        let rows = tensor.shape.first().copied().unwrap_or(0);
        let bytes = (rows + 1) * 4 + outlier_count * 4 + outlier_count * 2;
        let start = header.outlier_offset + offset;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("outlier region overflow for {name}"))?;
        if end > file_size {
            bail!("outlier region for tensor {name} exceeds file bounds");
        }
    }
    Ok(())
}

fn validate_codebook_regions(header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (name, codebook) in &metadata.codebooks {
        let start = header.codebook_offset + codebook.offset;
        let end = start
            .checked_add(codebook.size)
            .ok_or_else(|| anyhow!("codebook region overflow for {name}"))?;
        if end > file_size {
            bail!("codebook {name} exceeds file bounds");
        }
        ranges.push((start, end, name.as_str()));
    }
    for (name, scale) in &metadata.smoothquant_scales {
        let start = header.codebook_offset + scale.offset;
        let end = start
            .checked_add(scale.size)
            .ok_or_else(|| anyhow!("smoothquant region overflow for {name}"))?;
        if end > file_size {
            bail!("smoothquant scale {name} exceeds file bounds");
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    for window in ranges.windows(2) {
        if window[0].1 > window[1].0 {
            bail!("codebook regions overlap: {} and {}", window[0].2, window[1].2);
        }
    }
    Ok(())
}

fn validate_expert_dedup_regions(metadata: &Metadata, file_size: u64) -> Result<()> {
    let Some(dedup) = &metadata.expert_dedup else {
        return Ok(());
    };
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (name, tensor) in &metadata.tensors {
        let Some(offset) = tensor.dedup_correction_offset else {
            continue;
        };
        let count = tensor
            .dedup_correction_count
            .ok_or_else(|| anyhow!("tensor {name} is missing dedup_correction_count"))?;
        let rows = tensor.shape.first().copied().unwrap_or(0);
        let bytes = (rows + 1) * 4 + count * 4 + count * 2;
        let start = dedup.region_offset + offset;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("expert dedup region overflow for {name}"))?;
        if end > file_size {
            bail!("expert dedup region for tensor {name} exceeds file bounds");
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    for window in ranges.windows(2) {
        if window[0].1 > window[1].0 {
            bail!("expert dedup regions overlap: {} and {}", window[0].2, window[1].2);
        }
    }
    Ok(())
}

fn validate_speculative_region(header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    let Some(speculative) = &metadata.speculative_draft else {
        return Ok(());
    };
    let start = header
        .speculative_offset
        .checked_add(speculative.draft_offset)
        .ok_or_else(|| anyhow!("speculative draft offset overflow"))?;
    let end = start
        .checked_add(speculative.draft_bytes)
        .ok_or_else(|| anyhow!("speculative draft size overflow"))?;
    if end > file_size {
        bail!("speculative draft region exceeds file bounds");
    }
    Ok(())
}

fn is_lora_tensor(metadata: &Metadata, tensor: &TensorDescriptor) -> bool {
    metadata.lora.is_some() && tensor.target.is_some()
}

fn tensor_absolute_start(header: &BundleHeader, metadata: &Metadata, tensor: &TensorDescriptor) -> Result<u64> {
    if is_lora_tensor(metadata, tensor) {
        let lora = metadata
            .lora
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA tensor is present but metadata.lora is missing"))?;
        Ok(lora.region_offset + tensor.data_offset)
    } else {
        Ok(header.data_offset + tensor.data_offset)
    }
}

fn read_region_bytes(file: &mut File, start: u64, length: u64) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = vec![0_u8; length as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn checksum_file_region(file: &mut File, start: u64, length: u64) -> Result<String> {
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = length;
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut hasher = Xxh64::new(0);
    while remaining > 0 {
        let read_len = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..read_len])?;
        hasher.update(&buffer[..read_len]);
        remaining -= read_len as u64;
    }
    Ok(format!("{:016x}", hasher.digest()))
}

fn decode_f16_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn decode_bf16_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_mxq_tensor(tensor: &TensorDescriptor, payload: &[u8]) -> Result<Vec<f32>> {
    let bits = tensor.bits.ok_or_else(|| anyhow!("MXQ tensor missing bits"))? as usize;
    let group_size = tensor.group_size.ok_or_else(|| anyhow!("MXQ tensor missing group_size"))? as usize;
    let total_values = tensor.shape.iter().product::<u64>() as usize;
    let packed_group_bytes = (group_size * bits).div_ceil(8);
    let group_bytes = 2 + packed_group_bytes;
    let groups = total_values.div_ceil(group_size);
    if payload.len() < groups * group_bytes {
        bail!("MXQ tensor payload is too small");
    }

    let zero_point = 1_i32 << (bits - 1);
    let mut output = Vec::with_capacity(groups * group_size);
    for group in 0..groups {
        let start = group * group_bytes;
        let scale_bits = u16::from_le_bytes([payload[start], payload[start + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let packed = &payload[start + 2..start + group_bytes];
        let codes = unpack_codes(packed, bits, group_size);
        for code in codes {
            output.push(((code as i32 - zero_point) as f32) * scale);
        }
    }
    output.truncate(total_values);
    Ok(output)
}

fn decode_nf_tensor(tensor: &TensorDescriptor, payload: &[u8]) -> Result<Vec<f32>> {
    let bits = match tensor.dtype.as_str() {
        "axon_nf2" => 2,
        "axon_nf3" => 3,
        other => bail!("unsupported NF tensor dtype {other}"),
    };
    let group_size = tensor.group_size.ok_or_else(|| anyhow!("NF tensor missing group_size"))? as usize;
    let total_values = tensor.shape.iter().product::<u64>() as usize;
    let packed_group_bytes = (group_size * bits).div_ceil(8);
    let group_bytes = 2 + packed_group_bytes;
    let groups = total_values.div_ceil(group_size);
    if payload.len() < groups * group_bytes {
        bail!("NF tensor payload is too small");
    }

    let mut output = Vec::with_capacity(groups * group_size);
    for group in 0..groups {
        let start = group * group_bytes;
        let scale_bits = u16::from_le_bytes([payload[start], payload[start + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let packed = &payload[start + 2..start + group_bytes];
        let codes = unpack_codes(packed, bits, group_size);
        for code in codes {
            output.push(scale * nf_code_to_value(bits, code)?);
        }
    }
    output.truncate(total_values);
    Ok(output)
}

fn nf_code_to_value(bits: usize, code: u8) -> Result<f32> {
    match bits {
        2 => Ok(match code.min(3) {
            0 => -1.0,
            1 => -0.3333,
            2 => 0.3333,
            _ => 1.0,
        }),
        3 => Ok(match code.min(7) {
            0 => -1.0,
            1 => -0.5774,
            2 => -0.3333,
            3 => -0.1111,
            4 => 0.1111,
            5 => 0.3333,
            6 => 0.5774,
            _ => 1.0,
        }),
        _ => bail!("unsupported NF bit width {bits}"),
    }
}

fn unpack_codes(packed: &[u8], bits: usize, count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    let mask = ((1_u32 << bits) - 1) as u8;
    let mut bit_cursor = 0_usize;
    while values.len() < count {
        let byte_index = bit_cursor / 8;
        let shift = bit_cursor % 8;
        let mut word = packed.get(byte_index).copied().unwrap_or(0) as u32;
        if byte_index + 1 < packed.len() {
            word |= (packed[byte_index + 1] as u32) << 8;
        }
        if byte_index + 2 < packed.len() {
            word |= (packed[byte_index + 2] as u32) << 16;
        }
        values.push(((word >> shift) as u8) & mask);
        bit_cursor += bits;
    }
    values
}

fn decode_vq_tensor(
    file: &mut File,
    header: &BundleHeader,
    metadata: &Metadata,
    tensor: &TensorDescriptor,
    codes: &[u8],
) -> Result<Vec<f32>> {
    let codebook_id = tensor
        .codebook_id
        .as_deref()
        .ok_or_else(|| anyhow!("VQ tensor missing codebook_id"))?;
    let descriptor = metadata
        .codebooks
        .get(codebook_id)
        .ok_or_else(|| anyhow!("missing codebook {codebook_id}"))?;
    let codebook_bytes = read_region_bytes(file, header.codebook_offset + descriptor.offset, descriptor.size)?;
    let centers = decode_f16_bytes(&codebook_bytes);
    let entries = descriptor.entries as usize;
    let dim = descriptor.dim as usize;
    if entries * dim != centers.len() {
        bail!("codebook {codebook_id} has inconsistent size");
    }
    let vq_dim = tensor.vq_dim.ok_or_else(|| anyhow!("VQ tensor missing vq_dim"))? as usize;
    if dim != vq_dim {
        bail!("VQ tensor dimension mismatch");
    }
    let vector_count = tensor.shape.iter().product::<u64>() as usize / vq_dim;
    if codes.len() < vector_count {
        bail!("VQ tensor payload is too small");
    }
    let mut output = Vec::with_capacity(vector_count * vq_dim);
    for code in &codes[..vector_count] {
        let center_index = (*code as usize).min(entries.saturating_sub(1));
        let start = center_index * vq_dim;
        output.extend_from_slice(&centers[start..start + vq_dim]);
    }
    Ok(output)
}

fn apply_outliers(file: &mut File, header: &BundleHeader, tensor: &TensorDescriptor, output: &mut [f32]) -> Result<()> {
    let Some(offset) = tensor.outlier_indices_offset else {
        return Ok(());
    };
    let outlier_count = tensor.outlier_count.unwrap_or(0) as usize;
    if outlier_count == 0 {
        return Ok(());
    }
    let rows = tensor.shape.first().copied().unwrap_or(0) as usize;
    let cols = if rows == 0 { 0 } else { output.len() / rows };
    let base = header.outlier_offset + offset;
    let row_ptr_bytes = read_region_bytes(file, base, ((rows + 1) * 4) as u64)?;
    let row_ptr: Vec<u32> = row_ptr_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let col_idx_base = base + ((rows + 1) * 4) as u64;
    let col_idx_bytes = read_region_bytes(file, col_idx_base, (outlier_count * 4) as u64)?;
    let col_idx: Vec<u32> = col_idx_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let value_base = col_idx_base + (outlier_count * 4) as u64;
    let value_bytes = read_region_bytes(file, value_base, (outlier_count * 2) as u64)?;
    let values = decode_f16_bytes(&value_bytes);
    for row in 0..rows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for index in start..end {
            let col = col_idx[index] as usize;
            if col < cols {
                output[row * cols + col] += values[index];
            }
        }
    }
    Ok(())
}

fn apply_dedup_corrections(
    file: &mut File,
    metadata: &Metadata,
    tensor: &TensorDescriptor,
    output: &mut [f32],
) -> Result<()> {
    let Some(dedup) = &metadata.expert_dedup else {
        return Ok(());
    };
    let Some(offset) = tensor.dedup_correction_offset else {
        return Ok(());
    };
    let count = tensor.dedup_correction_count.unwrap_or(0) as usize;
    if count == 0 {
        return Ok(());
    }
    let rows = tensor.shape.first().copied().unwrap_or(0) as usize;
    let cols = if rows == 0 { 0 } else { output.len() / rows };
    let base = dedup.region_offset + offset;
    let row_ptr_bytes = read_region_bytes(file, base, ((rows + 1) * 4) as u64)?;
    let row_ptr: Vec<u32> = row_ptr_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let col_idx_base = base + ((rows + 1) * 4) as u64;
    let col_idx_bytes = read_region_bytes(file, col_idx_base, (count * 4) as u64)?;
    let col_idx: Vec<u32> = col_idx_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let value_base = col_idx_base + (count * 4) as u64;
    let value_bytes = read_region_bytes(file, value_base, (count * 2) as u64)?;
    let values = decode_f16_bytes(&value_bytes);
    for row in 0..rows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for index in start..end {
            let col = col_idx[index] as usize;
            if col < cols {
                output[row * cols + col] += values[index];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn align64(value: u64) -> u64 {
        if value % 64 == 0 {
            value
        } else {
            value + (64 - (value % 64))
        }
    }

    fn write_test_bundle(dir: &Path) {
        let mxq_payload = {
            let scale = f16::from_f32(2.0 / 7.0).to_bits().to_le_bytes();
            let packed = [0x4c_u8, 0xf8_u8];
            [scale.as_slice(), packed.as_slice()].concat()
        };
        let vq_codes = [0_u8];
        let codebook: Vec<u8> = {
            let mut out = Vec::new();
            let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for _ in 0..2 {
                for value in values {
                    out.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
                }
            }
            out
        };
        let mxq_checksum = format!("{:016x}", xxhash_rust::xxh64::xxh64(&mxq_payload, 0));
        let vq_checksum = format!("{:016x}", xxhash_rust::xxh64::xxh64(&vq_codes, 0));
        let metadata = serde_json::json!({
            "format": "axon",
            "version": "2.0.0-draft",
            "task": "causal_lm",
            "architecture": "llama",
            "model_family": "toy-llama",
            "model": {
                "hidden_dim": 8,
                "intermediate_dim": 16,
                "num_layers": 1,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 8,
                "context_length": 128,
                "vocab_size": 16,
                "total_parameter_count": 12,
                "active_parameter_count": 12,
                "rope": {"type": "rope", "theta": 10000.0, "scaling": null},
                "moe": null
            },
            "runtime": {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "default_generation": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_new_tokens": 16
                }
            },
            "source": {
                "format": "hf_safetensors",
                "identifier": "/tmp/source",
                "conversion_tool": "axon-pack 2.0.0",
                "conversion_time_utc": "2026-03-21T00:00:00Z"
            },
            "avg_bits_per_weight": 4.0,
            "quant_method": "axon-v2-mixed",
            "calibration": null,
            "tensors": {
                "weight": {
                    "shape": [4],
                    "dtype": "axon_mxq",
                    "bits": 4,
                    "group_size": 4,
                    "source_tensor_name": "weight",
                    "data_offset": 0,
                    "data_bytes": 4,
                    "scale_interleaved": true,
                    "outlier_indices_offset": null,
                    "outlier_count": null,
                    "sensitivity_score": 0.68,
                    "stream_order": 0,
                    "codebook_id": null,
                    "vq_dim": null,
                    "checksum_xxh64": mxq_checksum
                },
                "embed": {
                    "shape": [1, 8],
                    "dtype": "axon_vq",
                    "bits": null,
                    "group_size": null,
                    "source_tensor_name": "embed",
                    "data_offset": 64,
                    "data_bytes": 1,
                    "scale_interleaved": false,
                    "outlier_indices_offset": null,
                    "outlier_count": null,
                    "sensitivity_score": 0.95,
                    "stream_order": 1,
                    "codebook_id": "embed-codebook",
                    "vq_dim": 8,
                    "checksum_xxh64": vq_checksum
                }
            },
            "codebooks": {
                "embed-codebook": {
                    "offset": 0,
                    "entries": 2,
                    "dim": 8,
                    "dtype": "fp16",
                    "size": codebook.len()
                }
            },
            "hw_hints": {
                "cuda_sm80": {"kernel": "axonal_cuda_mxq_sm80"},
                "cpu_avx512_vnni": {"kernel": "axonal_cpu_mxq_avx512"}
            }
        });
        let metadata_bytes = serde_json::to_vec(&metadata).unwrap();
        let data_offset = align64(64 + metadata_bytes.len() as u64);
        let outlier_offset = 0_u64;
        let codebook_offset = align64(data_offset + 65);
        let mut header = [0_u8; 64];
        header[0..4].copy_from_slice(b"AXON");
        header[4..6].copy_from_slice(&2_u16.to_le_bytes());
        header[6..8].copy_from_slice(&0_u16.to_le_bytes());
        header[8..12].copy_from_slice(&(FLAG_HAS_CHECKSUMS | 1 | 4 | 8 | 32).to_le_bytes());
        header[12..16].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[16..24].copy_from_slice(&data_offset.to_le_bytes());
        header[24..32].copy_from_slice(&outlier_offset.to_le_bytes());
        header[32..40].copy_from_slice(&codebook_offset.to_le_bytes());

        fs::create_dir_all(dir).unwrap();
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "manifest_version": "2.0.0-draft",
                "task": "causal_lm",
                "model_file": "model.axon"
            }))
            .unwrap(),
        )
        .unwrap();

        let mut file = File::create(dir.join("model.axon")).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&metadata_bytes).unwrap();
        file.write_all(&vec![0_u8; (data_offset - (64 + metadata_bytes.len() as u64)) as usize]).unwrap();
        file.write_all(&mxq_payload).unwrap();
        file.write_all(&vec![0_u8; 60]).unwrap();
        file.write_all(&vq_codes).unwrap();
        let current = file.stream_position().unwrap();
        file.write_all(&vec![0_u8; (codebook_offset - current) as usize]).unwrap();
        file.write_all(&codebook).unwrap();
    }

    #[test]
    fn loads_valid_bundle() {
        let temp = tempdir().unwrap();
        write_test_bundle(temp.path());
        let bundle = load_bundle(temp.path()).unwrap();
        assert_eq!(bundle.metadata.model_family, "toy-llama");
        assert_eq!(bundle.metadata.tensors.len(), 2);
    }

    #[test]
    fn scans_registry() {
        let temp = tempdir().unwrap();
        let bundle_dir = temp.path().join("toy-llama-axon");
        write_test_bundle(&bundle_dir);
        let models = scan_registry(temp.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "toy-llama-axon");
    }

    #[test]
    fn scans_missing_registry_as_empty() {
        let temp = tempdir().unwrap();
        let models = scan_registry(&temp.path().join("missing-model-library")).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn decodes_quantized_and_vq_tensors() {
        let temp = tempdir().unwrap();
        write_test_bundle(temp.path());
        let mxq = decode_named_tensor(temp.path(), "weight").unwrap();
        assert_eq!(mxq.len(), 4);
        assert!(mxq[0] > 1.0 && mxq[0] < 1.2);
        let vq = decode_named_tensor(temp.path(), "embed").unwrap();
        assert_eq!(vq.len(), 8);
        assert_eq!(vq[0], 1.0);
        assert_eq!(vq[7], 8.0);
    }

    #[test]
    fn decodes_nf_tensor_payloads() {
        let nf2 = TensorDescriptor {
            shape: vec![4],
            dtype: "axon_nf2".to_string(),
            bits: Some(2),
            group_size: Some(4),
            source_tensor_name: None,
            data_offset: 0,
            data_bytes: 3,
            scale_interleaved: true,
            outlier_indices_offset: None,
            outlier_count: None,
            sensitivity_score: None,
            stream_order: 0,
            per_head_bits: None,
            nf_scale_fp16: true,
            smoothquant_scale: None,
            prefetch_priority: None,
            codebook_id: None,
            vq_dim: None,
            dedup_canonical: None,
            dedup_correction_offset: None,
            dedup_correction_count: None,
            lora_rank: None,
            lora_alpha: None,
            target: None,
            checksum_xxh64: None,
        };
        let mut nf2_payload = Vec::new();
        nf2_payload.extend_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        nf2_payload.push(0xE4);
        let decoded_nf2 = decode_nf_tensor(&nf2, &nf2_payload).unwrap();
        assert_eq!(decoded_nf2.len(), 4);
        assert!((decoded_nf2[0] + 2.0).abs() < 1e-4);
        assert!((decoded_nf2[1] + 0.6666).abs() < 5e-3);
        assert!((decoded_nf2[2] - 0.6666).abs() < 5e-3);
        assert!((decoded_nf2[3] - 2.0).abs() < 1e-4);

        let nf3 = TensorDescriptor {
            shape: vec![8],
            dtype: "axon_nf3".to_string(),
            bits: Some(3),
            group_size: Some(8),
            data_bytes: 5,
            ..nf2
        };
        let mut nf3_payload = Vec::new();
        nf3_payload.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        nf3_payload.extend_from_slice(&[0x88, 0x06, 0x00]);
        let decoded_nf3 = decode_nf_tensor(&nf3, &nf3_payload).unwrap();
        assert_eq!(decoded_nf3.len(), 8);
        assert!((decoded_nf3[0] + 1.0).abs() < 1e-4);
        assert!((decoded_nf3[1] + 0.5774).abs() < 5e-3);
        assert!((decoded_nf3[2] + 0.3333).abs() < 5e-3);
        assert!((decoded_nf3[3] + 0.1111).abs() < 5e-3);
    }
}
