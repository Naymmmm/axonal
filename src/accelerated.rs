use crate::{
    BundleHeader, BundleSummary, GenerationPreview, RunOptions, RuntimeProcessInfo, TensorDescriptor, backend,
    bundle_stats, load_bundle, resolve_lora_base_bundle,
};
use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use half::{bf16, f16};
use minijinja::{Environment, context};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

static CUDA_RUNTIME_CACHE: LazyLock<Mutex<HashMap<String, Arc<Mutex<CachedCudaRuntime>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct CacheAcquisition {
    runtime: Arc<Mutex<CachedCudaRuntime>>,
    note: Option<String>,
}

enum CachedModel {
    Qwen2(candle_transformers::models::qwen2::ModelForCausalLM),
    Llama {
        model: candle_transformers::models::llama::Llama,
        config: candle_transformers::models::llama::Config,
    },
}

struct CachedCudaRuntime {
    bundle_dir: PathBuf,
    summary: BundleSummary,
    device: Device,
    device_name: String,
    dtype: DType,
    tokenizer: Tokenizer,
    prompt_template: Option<String>,
    model: CachedModel,
    loaded_at_unix: u64,
    last_used_unix: u64,
    request_count: u64,
    active_requests: usize,
}

pub fn cached_runtime_processes() -> Vec<RuntimeProcessInfo> {
    let mut processes = CUDA_RUNTIME_CACHE
        .lock()
        .expect("cuda runtime cache lock poisoned")
        .values()
        .filter_map(|runtime| runtime.lock().ok().map(|runtime| runtime.process_info()))
        .collect::<Vec<_>>();
    processes.sort_by(|left, right| {
        right
            .last_used_unix
            .cmp(&left.last_used_unix)
            .then_with(|| left.name.cmp(&right.name))
    });
    processes
}

pub fn try_run_model(
    bundle_dir: &Path,
    prompt: &str,
    backend: &backend::BackendSelection,
    options: &RunOptions,
) -> Result<Option<GenerationPreview>> {
    if backend.backend != "cuda" {
        return Ok(None);
    }

    let cached = acquire_cached_runtime(bundle_dir)?;
    let mut runtime = cached.runtime.lock().expect("cached cuda runtime lock poisoned");
    runtime.last_used_unix = now_unix_secs();
    runtime.request_count += 1;
    runtime.active_requests += 1;
    let plan = GenerationPlan::resolve(&runtime.summary, options);
    let (prompt_text, add_special_tokens, prompt_note) = runtime.render_prompt(prompt, options.raw_prompt)?;
    let mut input_ids = runtime
        .tokenizer
        .encode(prompt_text, add_special_tokens)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?
        .get_ids()
        .to_vec();
    if input_ids.is_empty() {
        if let Some(bos) = runtime.summary.metadata.runtime.bos_token_id {
            input_ids.push(bos as u32);
        } else {
            bail!("prompt encoded to no tokens and no bos_token_id is configured");
        }
    }

    let generation_result = runtime.generate_ids(&input_ids, &plan);
    runtime.active_requests = runtime.active_requests.saturating_sub(1);
    runtime.last_used_unix = now_unix_secs();
    let generated_ids = generation_result?;

    let response = runtime
        .tokenizer
        .decode(&generated_ids, true)
        .map_err(|error| anyhow!("failed to decode output tokens: {error}"))?;
    let eos = runtime.summary.metadata.runtime.eos_token_id.map(|id| id as u32);
    let done_reason = if generated_ids.last().copied() == eos {
        "stop"
    } else {
        "length"
    };
    let mut message = format!(
        "Generated {} token(s) with {} backend using {}.",
        generated_ids.len(),
        backend.backend,
        backend.kernel
    );
    if let Some(note) = cached.note {
        message.push_str(&format!(" ({note})"));
    }
    if let Some(note) = prompt_note {
        message.push_str(&format!(" ({note})"));
    }
    Ok(Some(GenerationPreview {
        model: runtime.bundle_dir.display().to_string(),
        prompt: prompt.to_string(),
        response,
        done: true,
        done_reason: done_reason.to_string(),
        message,
        backend: backend.clone(),
        stats: bundle_stats(&runtime.summary),
    }))
}

fn acquire_cached_runtime(bundle_dir: &Path) -> Result<CacheAcquisition> {
    let cache_key = bundle_cache_key(bundle_dir)?;
    if let Some(runtime) = CUDA_RUNTIME_CACHE
        .lock()
        .expect("cuda runtime cache lock poisoned")
        .get(&cache_key)
        .cloned()
    {
        return Ok(CacheAcquisition {
            runtime,
            note: Some("reused warm cuda cache".to_string()),
        });
    }

    let started = Instant::now();
    let runtime = Arc::new(Mutex::new(CachedCudaRuntime::load(bundle_dir)?));
    let load_note = format!("loaded model into cuda cache in {}", format_duration(started.elapsed()));
    CUDA_RUNTIME_CACHE
        .lock()
        .expect("cuda runtime cache lock poisoned")
        .insert(cache_key, runtime.clone());
    Ok(CacheAcquisition {
        runtime,
        note: Some(load_note),
    })
}

fn bundle_cache_key(bundle_dir: &Path) -> Result<String> {
    let canonical = fs::canonicalize(bundle_dir).unwrap_or_else(|_| bundle_dir.to_path_buf());
    let summary = load_bundle(&canonical)?;
    let mut parts = vec![bundle_cache_component(&canonical)?];
    if let Some(lora) = summary.metadata.lora.as_ref() {
        let base_bundle = resolve_lora_base_bundle(&canonical, lora)?;
        parts.push(bundle_cache_component(&base_bundle)?);
    }
    Ok(parts.join("|"))
}

fn bundle_cache_component(bundle_dir: &Path) -> Result<String> {
    let canonical = fs::canonicalize(bundle_dir).unwrap_or_else(|_| bundle_dir.to_path_buf());
    let manifest_path = canonical.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: crate::BundleManifest =
        serde_json::from_str(&manifest_text).context("failed to parse manifest.json")?;
    let model_path = canonical.join(&manifest.model_file);
    let metadata = fs::metadata(&model_path)
        .with_context(|| format!("failed to stat {}", model_path.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    Ok(format!(
        "{}:{}:{}",
        canonical.display(),
        metadata.len(),
        modified
    ))
}

fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn build_cached_model(
    architecture: &str,
    config_text: &str,
    weights: &HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<CachedModel> {
    let vb = VarBuilder::from_tensors(weights.clone(), dtype, device);
    match architecture {
        "Qwen2ForCausalLM" => {
            let config: candle_transformers::models::qwen2::Config =
                serde_json::from_str(config_text).context("failed to parse qwen2 config")?;
            let model = candle_transformers::models::qwen2::ModelForCausalLM::new(&config, vb)
                .context("failed to build qwen2 CUDA model")?;
            Ok(CachedModel::Qwen2(model))
        }
        "LlamaForCausalLM" => {
            let llama_cfg: candle_transformers::models::llama::LlamaConfig =
                serde_json::from_str(config_text).context("failed to parse llama config")?;
            let config = llama_cfg.into_config(false);
            let model = candle_transformers::models::llama::Llama::load(vb, &config)
                .context("failed to build llama CUDA model")?;
            Ok(CachedModel::Llama { model, config })
        }
        other => Err(anyhow!("CUDA execution is not implemented for architecture {other}")),
    }
}

impl CachedCudaRuntime {
    fn load(bundle_dir: &Path) -> Result<Self> {
        let loaded = LoadedBundle::load(bundle_dir)?;
        let device = Device::new_cuda(0).map_err(|error| anyhow!("failed to initialize CUDA device: {error}"))?;
        let dtype = preferred_candle_dtype(&device);
        let tokenizer = load_tokenizer(&loaded.summary, &loaded.bundle_dir)?;
        let prompt_template = load_prompt_template(&loaded.summary, &loaded.bundle_dir)?;
        let config_path = loaded
            .summary
            .manifest
            .config_file
            .as_ref()
            .ok_or_else(|| anyhow!("bundle does not declare config.json"))?;
        let config_text = fs::read_to_string(loaded.bundle_dir.join(config_path))
            .with_context(|| format!("failed to read {}", loaded.bundle_dir.join(config_path).display()))?;
        let weights = load_candle_tensors(&loaded, &device)?;
        let model = build_cached_model(
            &loaded.summary.metadata.architecture,
            &config_text,
            &weights,
            dtype,
            &device,
        )?;
        let device_name = backend::select_backend(&loaded.summary, Some("cuda")).device;
        let now = now_unix_secs();
        Ok(Self {
            bundle_dir: loaded.bundle_dir,
            summary: loaded.summary,
            device,
            device_name,
            dtype,
            tokenizer,
            prompt_template,
            model,
            loaded_at_unix: now,
            last_used_unix: now,
            request_count: 0,
            active_requests: 0,
        })
    }

    fn render_prompt(&self, prompt: &str, raw_prompt: bool) -> Result<(String, bool, Option<String>)> {
        render_prompt_from_template(self.prompt_template.as_deref(), prompt, raw_prompt)
    }

    fn generate_ids(&mut self, input_ids: &[u32], plan: &GenerationPlan) -> Result<Vec<u32>> {
        let eos = self.summary.metadata.runtime.eos_token_id.map(|id| id as u32);
        match &mut self.model {
            CachedModel::Qwen2(model) => Self::generate_with_qwen2(model, &self.device, eos, input_ids, plan),
            CachedModel::Llama { model, config } => {
                Self::generate_with_llama(model, config, &self.device, self.dtype, eos, input_ids, plan)
            }
        }
    }

    fn generate_with_qwen2(
        model: &mut candle_transformers::models::qwen2::ModelForCausalLM,
        device: &Device,
        eos: Option<u32>,
        input_ids: &[u32],
        plan: &GenerationPlan,
    ) -> Result<Vec<u32>> {
        model.clear_kv_cache();
        let input = Tensor::from_slice(input_ids, (1, input_ids.len()), device)?;
        let mut logits = model.forward(&input, 0)?;
        let mut sampler = plan.logits_processor();
        let mut generated = Vec::new();
        let mut position = input_ids.len();

        for _ in 0..plan.max_new_tokens {
            let next = sampler.sample(&logits.i((0, 0))?)?;
            generated.push(next);
            if Some(next) == eos {
                break;
            }
            let token = Tensor::from_slice(&[next], (1, 1), device)?;
            logits = model.forward(&token, position)?;
            position += 1;
        }
        Ok(generated)
    }

    fn generate_with_llama(
        model: &candle_transformers::models::llama::Llama,
        config: &candle_transformers::models::llama::Config,
        device: &Device,
        dtype: DType,
        eos: Option<u32>,
        input_ids: &[u32],
        plan: &GenerationPlan,
    ) -> Result<Vec<u32>> {
        let mut cache = candle_transformers::models::llama::Cache::new(true, dtype, config, device)
            .context("failed to initialize llama cache")?;
        let input = Tensor::from_slice(input_ids, (1, input_ids.len()), device)?;
        let mut logits = model.forward(&input, 0, &mut cache)?;
        let mut sampler = plan.logits_processor();
        let mut generated = Vec::new();
        let mut position = input_ids.len();

        for _ in 0..plan.max_new_tokens {
            let next = sampler.sample(&logits.i(0)?)?;
            generated.push(next);
            if Some(next) == eos {
                break;
            }
            let token = Tensor::from_slice(&[next], (1, 1), device)?;
            logits = model.forward(&token, position, &mut cache)?;
            position += 1;
        }
        Ok(generated)
    }

    fn process_info(&self) -> RuntimeProcessInfo {
        RuntimeProcessInfo {
            name: self
                .bundle_dir
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string()),
            path: self.bundle_dir.display().to_string(),
            architecture: self.summary.metadata.architecture.clone(),
            model_family: self.summary.metadata.model_family.clone(),
            backend: "cuda".to_string(),
            device: self.device_name.clone(),
            file_size: self.summary.file_size,
            loaded_at_unix: self.loaded_at_unix,
            last_used_unix: self.last_used_unix,
            request_count: self.request_count,
            active_requests: self.active_requests,
            state: if self.active_requests > 0 {
                "running".to_string()
            } else {
                "loaded".to_string()
            },
        }
    }
}

#[derive(Clone)]
struct LoadedBundle {
    bundle_dir: PathBuf,
    summary: BundleSummary,
    header: BundleHeader,
    bytes: Vec<u8>,
    lora: Option<LoraOverlay>,
}

#[derive(Clone)]
struct LoraOverlay {
    summary: BundleSummary,
    header: BundleHeader,
    bytes: Vec<u8>,
    targets: HashMap<String, LoraTarget>,
}

#[derive(Clone)]
struct LoraTarget {
    a_name: String,
    b_name: String,
    scale: f32,
}

#[derive(Default)]
struct PendingLoraTarget {
    a_name: Option<String>,
    b_name: Option<String>,
    scale: Option<f32>,
}

impl LoadedBundle {
    fn load(bundle_dir: &Path) -> Result<Self> {
        let raw_summary = load_bundle(bundle_dir)?;
        let (summary, model_bundle_dir, model_file, lora) = if let Some(lora) = raw_summary.metadata.lora.clone() {
            let base_bundle_dir = resolve_lora_base_bundle(bundle_dir, &lora)?;
            let base_summary = load_bundle(&base_bundle_dir)?;
            let model_file = base_summary.manifest.model_file.clone();
            let mut merged_summary = base_summary;
            merged_summary.manifest = raw_summary.manifest.clone();
            merged_summary.metadata.runtime = raw_summary.metadata.runtime.clone();
            merged_summary.metadata.source = raw_summary.metadata.source.clone();
            merged_summary.metadata.lora = raw_summary.metadata.lora.clone();
            merged_summary.file_size = raw_summary.file_size;
            merged_summary.flags = raw_summary.flags;
            (
                merged_summary,
                base_bundle_dir,
                model_file,
                Some(LoraOverlay::load(bundle_dir, raw_summary.clone())?),
            )
        } else {
            (
                raw_summary.clone(),
                bundle_dir.to_path_buf(),
                raw_summary.manifest.model_file.clone(),
                None,
            )
        };
        let model_path = model_bundle_dir.join(&model_file);
        let bytes = fs::read(&model_path).with_context(|| format!("failed to read {}", model_path.display()))?;
        let header = parse_header(&bytes)?;
        if let Some(overlay) = &lora {
            for target in overlay.targets.keys() {
                if !summary.metadata.tensors.contains_key(target) {
                    bail!("LoRA delta targets missing base tensor {target}");
                }
            }
        }
        Ok(Self {
            bundle_dir: bundle_dir.to_path_buf(),
            summary,
            header,
            bytes,
            lora,
        })
    }

    fn tensor_bytes(&self, tensor: &TensorDescriptor) -> Result<&[u8]> {
        let tensor = if let Some(canonical) = tensor.dedup_canonical.as_deref() {
            self.summary
                .metadata
                .tensors
                .get(canonical)
                .ok_or_else(|| anyhow!("missing canonical tensor {canonical}"))?
        } else {
            tensor
        };
        let base = if self.summary.metadata.lora.is_some() && tensor.target.is_some() {
            self.summary
                .metadata
                .lora
                .as_ref()
                .ok_or_else(|| anyhow!("LoRA tensor is present but lora metadata is missing"))?
                .region_offset
        } else {
            self.header.data_offset
        };
        let start = (base + tensor.data_offset) as usize;
        let end = start + tensor.data_bytes as usize;
        self.bytes
            .get(start..end)
            .ok_or_else(|| anyhow!("tensor payload is out of bounds"))
    }

    fn codebook_bytes(&self, codebook_id: &str) -> Result<&[u8]> {
        let descriptor = self
            .summary
            .metadata
            .codebooks
            .get(codebook_id)
            .ok_or_else(|| anyhow!("missing codebook {codebook_id}"))?;
        let start = (self.header.codebook_offset + descriptor.offset) as usize;
        let end = start + descriptor.size as usize;
        self.bytes
            .get(start..end)
            .ok_or_else(|| anyhow!("codebook payload is out of bounds"))
    }

    fn outlier_bytes(&self, tensor: &TensorDescriptor) -> Result<Option<&[u8]>> {
        let Some(offset) = tensor.outlier_indices_offset else {
            return Ok(None);
        };
        let outlier_count = tensor.outlier_count.unwrap_or(0) as usize;
        let rows = tensor.shape.first().copied().unwrap_or(0) as usize;
        let length = ((rows + 1) * 4) + (outlier_count * 4) + (outlier_count * 2);
        let start = (self.header.outlier_offset + offset) as usize;
        let end = start + length;
        Ok(self.bytes.get(start..end))
    }

    fn lora_target(&self, target_name: &str) -> Option<&LoraTarget> {
        self.lora.as_ref()?.targets.get(target_name)
    }

    fn decode_lora_matrix(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        self.lora
            .as_ref()
            .ok_or_else(|| anyhow!("missing LoRA overlay"))?
            .dense_matrix(name)
    }
}

impl LoraOverlay {
    fn load(bundle_dir: &Path, summary: BundleSummary) -> Result<Self> {
        let model_path = bundle_dir.join(&summary.manifest.model_file);
        let bytes = fs::read(&model_path).with_context(|| format!("failed to read {}", model_path.display()))?;
        let header = parse_header(&bytes)?;
        let targets = build_lora_targets(&summary)?;
        Ok(Self {
            summary,
            header,
            bytes,
            targets,
        })
    }

    fn tensor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.summary
            .metadata
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing LoRA tensor {name}"))
    }

    fn tensor_bytes(&self, tensor: &TensorDescriptor) -> Result<&[u8]> {
        let base = if self.summary.metadata.lora.is_some() && tensor.target.is_some() {
            self.summary
                .metadata
                .lora
                .as_ref()
                .ok_or_else(|| anyhow!("LoRA tensor is present but lora metadata is missing"))?
                .region_offset
        } else {
            self.header.data_offset
        };
        let start = (base + tensor.data_offset) as usize;
        let end = start + tensor.data_bytes as usize;
        self.bytes
            .get(start..end)
            .ok_or_else(|| anyhow!("LoRA tensor payload is out of bounds"))
    }

    fn dense_matrix(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        let tensor = self.tensor(name)?.clone();
        let shape: Vec<usize> = tensor.shape.iter().map(|dim| *dim as usize).collect();
        if shape.len() < 2 {
            bail!("LoRA tensor {name} is not a matrix");
        }
        let rows = shape[0];
        let cols = shape[1..].iter().product();
        let payload = self.tensor_bytes(&tensor)?;
        let values = match tensor.dtype.as_str() {
            "fp16" => decode_f16_vec(payload).into_iter().map(|value| value.to_f32()).collect(),
            "bf16" => decode_bf16_vec(payload).into_iter().map(|value| value.to_f32()).collect(),
            "fp32" => decode_f32_vec(payload),
            other => bail!("unsupported LoRA tensor dtype {other} for {name}"),
        };
        if values.len() != rows * cols {
            bail!("decoded LoRA tensor {name} has {} values, expected {}", values.len(), rows * cols);
        }
        Ok((values, rows, cols))
    }
}

fn build_lora_targets(summary: &BundleSummary) -> Result<HashMap<String, LoraTarget>> {
    let lora = summary
        .metadata
        .lora
        .as_ref()
        .ok_or_else(|| anyhow!("missing lora metadata"))?;
    let mut pending = HashMap::<String, PendingLoraTarget>::new();
    for (name, tensor) in &summary.metadata.tensors {
        let Some(target) = tensor.target.as_ref() else {
            continue;
        };
        let rank = tensor.lora_rank.unwrap_or(lora.rank).max(1);
        let alpha = tensor.lora_alpha.unwrap_or(lora.alpha);
        let entry = pending.entry(target.clone()).or_default();
        entry.scale.get_or_insert(alpha as f32 / rank as f32);
        if name.ends_with(".lora_A") {
            entry.a_name = Some(name.clone());
        } else if name.ends_with(".lora_B") {
            entry.b_name = Some(name.clone());
        } else if entry.a_name.is_none() {
            entry.a_name = Some(name.clone());
        } else if entry.b_name.is_none() {
            entry.b_name = Some(name.clone());
        }
    }

    let mut targets = HashMap::new();
    for (target, target_spec) in pending {
        let a_name = target_spec
            .a_name
            .ok_or_else(|| anyhow!("LoRA target {target} is missing its A tensor"))?;
        let b_name = target_spec
            .b_name
            .ok_or_else(|| anyhow!("LoRA target {target} is missing its B tensor"))?;
        targets.insert(
            target,
            LoraTarget {
                a_name,
                b_name,
                scale: target_spec.scale.unwrap_or(1.0),
            },
        );
    }
    Ok(targets)
}

#[derive(Clone)]
struct GenerationPlan {
    max_new_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    seed: u64,
}

impl GenerationPlan {
    fn resolve(summary: &BundleSummary, options: &RunOptions) -> Self {
        let defaults = &summary.metadata.runtime.default_generation;
        let max_new_tokens = options
            .max_tokens
            .unwrap_or(defaults.max_new_tokens as usize)
            .max(1);
        let temperature = options.temperature.unwrap_or(defaults.temperature);
        let top_p = options.top_p.unwrap_or(defaults.top_p);
        let top_k = options.top_k.unwrap_or(defaults.top_k as usize);
        let seed = options.seed.unwrap_or_else(default_seed);
        Self {
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        }
    }

    fn logits_processor(&self) -> LogitsProcessor {
        let sampling = if self.temperature <= 1e-7 {
            Sampling::ArgMax
        } else if self.top_k > 0 && self.top_p > 0.0 && self.top_p < 1.0 {
            Sampling::TopKThenTopP {
                k: self.top_k,
                p: self.top_p,
                temperature: self.temperature,
            }
        } else if self.top_k > 0 {
            Sampling::TopK {
                k: self.top_k,
                temperature: self.temperature,
            }
        } else if self.top_p > 0.0 && self.top_p < 1.0 {
            Sampling::TopP {
                p: self.top_p,
                temperature: self.temperature,
            }
        } else {
            Sampling::All {
                temperature: self.temperature,
            }
        };
        LogitsProcessor::from_sampling(self.seed, sampling)
    }
}

fn load_candle_tensors(loaded: &LoadedBundle, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    for (name, tensor) in &loaded.summary.metadata.tensors {
        tensors.insert(name.clone(), decode_tensor(loaded, name, tensor, device)?);
    }
    Ok(tensors)
}

fn decode_tensor(loaded: &LoadedBundle, name: &str, tensor: &TensorDescriptor, device: &Device) -> Result<Tensor> {
    let shape: Vec<usize> = tensor.shape.iter().map(|dim| *dim as usize).collect();
    let mut values = match tensor.dtype.as_str() {
        "fp16" => {
            decode_f16_vec(loaded.tensor_bytes(tensor)?)
                .into_iter()
                .map(|value| value.to_f32())
                .collect()
        }
        "bf16" => {
            decode_bf16_vec(loaded.tensor_bytes(tensor)?)
                .into_iter()
                .map(|value| value.to_f32())
                .collect()
        }
        "fp32" => {
            decode_f32_vec(loaded.tensor_bytes(tensor)?)
        }
        "axon_mxq" => {
            decode_mxq_dense(loaded, tensor)?
        }
        "axon_nf2" | "axon_nf3" => {
            decode_nf_dense(loaded, tensor)?
        }
        "axon_vq" => {
            decode_vq_dense(loaded, tensor)?
        }
        other => bail!("unsupported AXON tensor dtype {other}"),
    };
    if tensor.shape.len() >= 2 {
        apply_lora_dense_merge(loaded, name, tensor, &mut values)?;
    }
    Tensor::from_vec(values, shape, device).map_err(Into::into)
}

fn apply_lora_dense_merge(
    loaded: &LoadedBundle,
    name: &str,
    tensor: &TensorDescriptor,
    base_values: &mut [f32],
) -> Result<()> {
    let Some(target) = loaded.lora_target(name).cloned() else {
        return Ok(());
    };
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1..].iter().product::<u64>() as usize;
    if base_values.len() != rows * cols {
        bail!("LoRA merge shape mismatch for {name}");
    }
    let (a, a_rows, a_cols) = loaded.decode_lora_matrix(&target.a_name)?;
    let (b, b_rows, b_cols) = loaded.decode_lora_matrix(&target.b_name)?;
    if a_cols != cols || b_rows != rows || b_cols != a_rows {
        bail!(
            "LoRA factors are incompatible for {name}: base is {}x{}, A is {}x{}, B is {}x{}",
            rows,
            cols,
            a_rows,
            a_cols,
            b_rows,
            b_cols
        );
    }
    for row in 0..rows {
        let out_start = row * cols;
        let out_end = out_start + cols;
        let out_slice = &mut base_values[out_start..out_end];
        for rank_index in 0..b_cols {
            let coeff = b[row * b_cols + rank_index] * target.scale;
            if coeff == 0.0 {
                continue;
            }
            let a_start = rank_index * cols;
            let a_end = a_start + cols;
            for (value, update) in out_slice.iter_mut().zip(a[a_start..a_end].iter()) {
                *value += coeff * *update;
            }
        }
    }
    Ok(())
}

fn decode_mxq_dense(loaded: &LoadedBundle, tensor: &TensorDescriptor) -> Result<Vec<f32>> {
    let element_count = tensor.shape.iter().product::<u64>() as usize;
    let bits = tensor.bits.ok_or_else(|| anyhow!("MXQ tensor missing bits"))? as usize;
    let group_size = tensor.group_size.ok_or_else(|| anyhow!("MXQ tensor missing group_size"))? as usize;
    let payload = loaded.tensor_bytes(tensor)?;
    let packed_group_bytes = (group_size * bits).div_ceil(8);
    let group_bytes = 2 + packed_group_bytes;
    let zero_point = 1_i32 << (bits - 1);
    let mut values = vec![0.0_f32; element_count];
    let group_count = element_count.div_ceil(group_size);

    for group_index in 0..group_count {
        let start = group_index * group_size;
        let take = (element_count - start).min(group_size);
        let group_offset = group_index * group_bytes;
        let scale_bits = u16::from_le_bytes([payload[group_offset], payload[group_offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let packed = &payload[group_offset + 2..group_offset + group_bytes];
        let codes = unpack_codes(packed, bits, group_size);
        for index in 0..take {
            values[start + index] = ((codes[index] as i32 - zero_point) as f32) * scale;
        }
    }

    if let Some(outlier_bytes) = loaded.outlier_bytes(tensor)? {
        apply_outliers_to_dense(tensor, outlier_bytes, &mut values)?;
    }
    Ok(values)
}

fn decode_vq_dense(loaded: &LoadedBundle, tensor: &TensorDescriptor) -> Result<Vec<f32>> {
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1..].iter().product::<u64>() as usize;
    let vq_dim = tensor.vq_dim.ok_or_else(|| anyhow!("VQ tensor missing vq_dim"))? as usize;
    let codebook_id = tensor
        .codebook_id
        .as_ref()
        .ok_or_else(|| anyhow!("VQ tensor missing codebook_id"))?;
    let centers = decode_f16_bytes(loaded.codebook_bytes(codebook_id)?);
    let codes = loaded.tensor_bytes(tensor)?;
    let segments = cols / vq_dim;
    let mut values = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let row_codes = &codes[row * segments..(row + 1) * segments];
        for (segment, &code) in row_codes.iter().enumerate() {
            let center = &centers[(code as usize) * vq_dim..(code as usize + 1) * vq_dim];
            let start = row * cols + segment * vq_dim;
            values[start..start + vq_dim].copy_from_slice(center);
        }
    }
    if let Some(outlier_bytes) = loaded.outlier_bytes(tensor)? {
        apply_outliers_to_dense(tensor, outlier_bytes, &mut values)?;
    }
    Ok(values)
}

fn decode_nf_dense(loaded: &LoadedBundle, tensor: &TensorDescriptor) -> Result<Vec<f32>> {
    let element_count = tensor.shape.iter().product::<u64>() as usize;
    let bits = match tensor.dtype.as_str() {
        "axon_nf2" => 2,
        "axon_nf3" => 3,
        other => bail!("unsupported NF tensor dtype {other}"),
    };
    let group_size = tensor.group_size.ok_or_else(|| anyhow!("NF tensor missing group_size"))? as usize;
    let payload = loaded.tensor_bytes(tensor)?;
    let packed_group_bytes = (group_size * bits).div_ceil(8);
    let group_bytes = 2 + packed_group_bytes;
    let mut values = vec![0.0_f32; element_count];
    let group_count = element_count.div_ceil(group_size);

    for group_index in 0..group_count {
        let start = group_index * group_size;
        let take = (element_count - start).min(group_size);
        let group_offset = group_index * group_bytes;
        let scale_bits = u16::from_le_bytes([payload[group_offset], payload[group_offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let packed = &payload[group_offset + 2..group_offset + group_bytes];
        let codes = unpack_codes(packed, bits, group_size);
        for index in 0..take {
            values[start + index] = scale * nf_code_to_value(bits, codes[index])?;
        }
    }

    if let Some(outlier_bytes) = loaded.outlier_bytes(tensor)? {
        apply_outliers_to_dense(tensor, outlier_bytes, &mut values)?;
    }
    Ok(values)
}

fn apply_outliers_to_dense(tensor: &TensorDescriptor, outlier_bytes: &[u8], values: &mut [f32]) -> Result<()> {
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1..].iter().product::<u64>() as usize;
    let outlier_count = tensor.outlier_count.unwrap_or(0) as usize;
    let row_ptr_bytes = &outlier_bytes[..(rows + 1) * 4];
    let row_ptr: Vec<u32> = row_ptr_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let col_idx_bytes = &outlier_bytes[(rows + 1) * 4..(rows + 1) * 4 + outlier_count * 4];
    let col_idx: Vec<u32> = col_idx_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let value_offset = (rows + 1) * 4 + outlier_count * 4;
    let outlier_values = decode_f16_bytes(&outlier_bytes[value_offset..]);
    for row in 0..rows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for index in start..end {
            let col = col_idx[index] as usize;
            if col < cols {
                values[row * cols + col] += outlier_values[index];
            }
        }
    }
    Ok(())
}

fn load_tokenizer(summary: &BundleSummary, bundle_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_manifest = summary
        .manifest
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow!("bundle does not declare tokenizer assets"))?;
    let tokenizer_file = tokenizer_manifest
        .files
        .iter()
        .find(|file| file.ends_with("tokenizer.json"))
        .ok_or_else(|| anyhow!("bundle does not include tokenizer.json; axonal currently requires a Hugging Face tokenizer.json bundle asset"))?;
    Tokenizer::from_file(bundle_dir.join(tokenizer_file))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))
}

#[derive(Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

fn load_prompt_template(summary: &BundleSummary, bundle_dir: &Path) -> Result<Option<String>> {
    let Some(tokenizer_manifest) = summary.manifest.tokenizer.as_ref() else {
        return Ok(None);
    };
    let Some(config_file) = tokenizer_manifest
        .files
        .iter()
        .find(|file| file.ends_with("tokenizer_config.json"))
    else {
        let template = tokenizer_manifest
            .files
            .iter()
            .find(|file| file.ends_with("chat_template.jinja"))
            .map(|file| bundle_dir.join(file))
            .map(|path| {
                fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))
            })
            .transpose()?;
        return Ok(template);
    };
    let config_path = bundle_dir.join(config_file);
    let config_text = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_json: serde_json::Value =
        serde_json::from_str(&config_text).context("failed to parse tokenizer_config.json")?;
    Ok(config_json
        .get("chat_template")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            tokenizer_manifest
                .files
                .iter()
                .find(|file| file.ends_with("chat_template.jinja"))
                .and_then(|file| fs::read_to_string(bundle_dir.join(file)).ok())
        }))
}

fn render_prompt_from_template(
    template: Option<&str>,
    prompt: &str,
    raw_prompt: bool,
) -> Result<(String, bool, Option<String>)> {
    if raw_prompt {
        return Ok((prompt.to_string(), true, None));
    }
    let Some(template) = template else {
        return Ok((prompt.to_string(), true, None));
    };

    let mut env = Environment::new();
    env.add_template("chat", template)
        .context("failed to compile chat template")?;
    let rendered = env
        .get_template("chat")?
        .render(context! {
            messages => vec![TemplateMessage {
                role: "user",
                content: prompt,
            }],
            add_generation_prompt => true,
            tools => Vec::<serde_json::Value>::new(),
        })
        .context("failed to render chat template")?;
    Ok((rendered, false, Some("applied chat template".to_string())))
}

fn preferred_candle_dtype(device: &Device) -> DType {
    match device {
        Device::Cpu => DType::F32,
        _ => DType::F16,
    }
}

fn parse_header(bytes: &[u8]) -> Result<BundleHeader> {
    if bytes.len() < 64 {
        bail!("AXON file is shorter than the fixed header");
    }
    let header = &bytes[..64];
    if header[0..4] != *b"AXON" {
        bail!("invalid AXON magic");
    }
    Ok(BundleHeader {
        version_major: u16::from_le_bytes(header[4..6].try_into().unwrap()),
        version_minor: u16::from_le_bytes(header[6..8].try_into().unwrap()),
        flags: u32::from_le_bytes(header[8..12].try_into().unwrap()),
        header_len: u32::from_le_bytes(header[12..16].try_into().unwrap()),
        data_offset: u64::from_le_bytes(header[16..24].try_into().unwrap()),
        outlier_offset: u64::from_le_bytes(header[24..32].try_into().unwrap()),
        codebook_offset: u64::from_le_bytes(header[32..40].try_into().unwrap()),
        tail_offset: u64::from_le_bytes(header[40..48].try_into().unwrap()),
        boot_cutoff: header[48],
        speculative_offset: u64::from_le_bytes(header[52..60].try_into().unwrap()),
    })
}

fn decode_f16_vec(bytes: &[u8]) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn decode_bf16_vec(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_f16_bytes(bytes: &[u8]) -> Vec<f32> {
    decode_f16_vec(bytes).into_iter().map(|value| value.to_f32()).collect()
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

fn default_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
