use crate::{
    BundleHeader, BundleSummary, GenerationPreview, RunOptions, TensorDescriptor, accelerated, backend,
    bundle_stats, load_bundle, resolve_lora_base_bundle,
};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use minijinja::{Environment, context};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

pub fn run_model(
    bundle_dir: &Path,
    prompt: &str,
    preferred_backend: Option<&str>,
    max_tokens: Option<usize>,
) -> Result<GenerationPreview> {
    run_model_with_options(
        bundle_dir,
        prompt,
        preferred_backend,
        RunOptions {
            max_tokens,
            ..RunOptions::default()
        },
    )
}

pub fn run_model_with_options(
    bundle_dir: &Path,
    prompt: &str,
    preferred_backend: Option<&str>,
    options: RunOptions,
) -> Result<GenerationPreview> {
    let loaded = LoadedBundle::load(bundle_dir)?;
    let selected_backend = backend::select_backend(&loaded.summary, preferred_backend);
    if selected_backend.backend == "cuda" {
        match accelerated::try_run_model(bundle_dir, prompt, &selected_backend, &options) {
            Ok(Some(preview)) => return Ok(preview),
            Ok(None) => {}
            Err(error) => {
                let cpu_backend = backend::cpu_backend(&loaded.summary);
                return run_manual_model(loaded, prompt, cpu_backend, Some(format!(
                    "requested cuda backend, falling back to CPU: {error:#}"
                )), &options);
            }
        }
    }
    let cpu_backend = backend::cpu_backend(&loaded.summary);
    run_manual_model(loaded, prompt, cpu_backend, None, &options)
}

fn run_manual_model(
    loaded: LoadedBundle,
    prompt: &str,
    backend: backend::BackendSelection,
    backend_note: Option<String>,
    options: &RunOptions,
) -> Result<GenerationPreview> {
    let mut runtime = RuntimeModel::new(loaded)?;
    let tokenizer = runtime.load_tokenizer()?;
    let (prompt_text, add_special_tokens, prompt_note) =
        render_prompt(&runtime.bundle.summary, &runtime.bundle.bundle_dir, prompt, options.raw_prompt)?;
    let plan = GenerationPlan::resolve(&runtime.bundle.summary, options);
    let mut input_ids = tokenizer
        .encode(prompt_text, add_special_tokens)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?
        .get_ids()
        .to_vec();
    if input_ids.is_empty() {
        if let Some(bos) = runtime.bundle.summary.metadata.runtime.bos_token_id {
            input_ids.push(bos as u32);
        } else {
            bail!("prompt encoded to no tokens and no bos_token_id is configured");
        }
    }
    let generated_ids = runtime.generate_ids_with_plan(&input_ids, &plan)?;
    let response = tokenizer
        .decode(&generated_ids, true)
        .map_err(|error| anyhow!("failed to decode output tokens: {error}"))?;
    let eos = runtime.bundle.summary.metadata.runtime.eos_token_id.map(|id| id as u32);
    let done_reason = if generated_ids.last().copied() == eos {
        "stop"
    } else {
        "length"
    };
    let mut notes = Vec::new();
    if let Some(note) = backend_note {
        notes.push(note);
    }
    if let Some(note) = prompt_note {
        notes.push(note);
    }
    Ok(GenerationPreview {
        model: runtime.bundle.bundle_dir.display().to_string(),
        prompt: prompt.to_string(),
        response,
        done: true,
        done_reason: done_reason.to_string(),
        message: format!(
            "Generated {} token(s) with {} backend using {}{}.",
            generated_ids.len(),
            backend.backend,
            backend.kernel,
            if notes.is_empty() {
                String::new()
            } else {
                format!(" ({})", notes.join("; "))
            }
        ),
        backend,
        stats: bundle_stats(&runtime.bundle.summary),
    })
}

pub fn generate_token_ids(
    bundle_dir: &Path,
    input_ids: &[u32],
    max_new_tokens: usize,
) -> Result<Vec<u32>> {
    let loaded = LoadedBundle::load(bundle_dir)?;
    let mut runtime = RuntimeModel::new(loaded)?;
    runtime.generate_ids_with_plan(
        input_ids,
        &GenerationPlan {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        },
    )
}

#[derive(Clone)]
struct LoadedBundle {
    bundle_dir: PathBuf,
    summary: BundleSummary,
    header: BundleHeader,
    bytes: Vec<u8>,
    decoder: DecoderSpec,
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

#[derive(Clone)]
struct LoraWeights {
    a: Vec<f32>,
    a_rows: usize,
    a_cols: usize,
    b: Vec<f32>,
    b_rows: usize,
    b_cols: usize,
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
        let decoder = DecoderSpec::from_summary(&summary)?;
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
            decoder,
            lora,
        })
    }

    fn tensor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.summary
            .metadata
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
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

    fn lora_target(&self, target_name: &str) -> Option<&LoraTarget> {
        self.lora.as_ref()?.targets.get(target_name)
    }

    fn decode_lora_matrix(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        self.lora
            .as_ref()
            .ok_or_else(|| anyhow!("missing LoRA overlay"))?
            .dense_matrix(name)
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
        let (rows, cols) = matrix_shape(&tensor)?;
        let payload = self.tensor_bytes(&tensor)?;
        let values = match tensor.dtype.as_str() {
            "fp16" => decode_f16_bytes(payload),
            "bf16" => decode_bf16_bytes(payload),
            "fp32" => decode_f32_bytes(payload),
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
        } else if self.temperature <= 1e-7 {
            Sampling::ArgMax
        } else {
            Sampling::All {
                temperature: self.temperature,
            }
        };
        LogitsProcessor::from_sampling(self.seed, sampling)
    }
}

#[derive(Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

fn render_prompt(
    summary: &BundleSummary,
    bundle_dir: &Path,
    prompt: &str,
    raw_prompt: bool,
) -> Result<(String, bool, Option<String>)> {
    if raw_prompt {
        return Ok((prompt.to_string(), true, None));
    }
    let Some(tokenizer_manifest) = summary.manifest.tokenizer.as_ref() else {
        return Ok((prompt.to_string(), true, None));
    };
    let Some(config_file) = tokenizer_manifest
        .files
        .iter()
        .find(|file| file.ends_with("tokenizer_config.json"))
    else {
        if let Some(template_file) = tokenizer_manifest
            .files
            .iter()
            .find(|file| file.ends_with("chat_template.jinja"))
        {
            let template_path = bundle_dir.join(template_file);
            let template = fs::read_to_string(&template_path)
                .with_context(|| format!("failed to read {}", template_path.display()))?;
            let mut env = Environment::new();
            env.add_template("chat", &template)
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
            return Ok((
                rendered,
                false,
                Some("applied chat template".to_string()),
            ));
        }
        return Ok((prompt.to_string(), true, None));
    };
    let config_path = bundle_dir.join(config_file);
    let config_text = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_json: serde_json::Value =
        serde_json::from_str(&config_text).context("failed to parse tokenizer_config.json")?;
    let template = config_json
        .get("chat_template")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            tokenizer_manifest
                .files
                .iter()
                .find(|file| file.ends_with("chat_template.jinja"))
                .and_then(|file| fs::read_to_string(bundle_dir.join(file)).ok())
        });
    let Some(template) = template else {
        return Ok((prompt.to_string(), true, None));
    };

    let mut env = Environment::new();
    env.add_template("chat", &template)
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
    Ok((
        rendered,
        false,
        Some("applied chat template".to_string()),
    ))
}

fn default_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[derive(Clone)]
struct DecoderSpec {
    embedding: String,
    final_norm: String,
    lm_head: String,
    layers: Vec<LayerSpec>,
}

#[derive(Clone)]
struct LayerSpec {
    input_norm: String,
    post_norm: String,
    q_proj: String,
    q_bias: Option<String>,
    k_proj: String,
    k_bias: Option<String>,
    v_proj: String,
    v_bias: Option<String>,
    o_proj: String,
    o_bias: Option<String>,
    gate_proj: String,
    gate_bias: Option<String>,
    up_proj: String,
    up_bias: Option<String>,
    down_proj: String,
    down_bias: Option<String>,
}

impl DecoderSpec {
    fn from_summary(summary: &BundleSummary) -> Result<Self> {
        let tensors = &summary.metadata.tensors;
        let embedding = find_required_tensor(
            tensors,
            &[
                "model.embed_tokens.weight",
                "tok_embeddings.weight",
                "transformer.wte.weight",
            ],
        )?;
        let final_norm = find_required_tensor(
            tensors,
            &["model.norm.weight", "transformer.norm.weight", "norm.weight"],
        )?;
        let lm_head = find_optional_tensor(tensors, &["lm_head.weight", "output.weight"])
            .unwrap_or_else(|| embedding.clone());

        let mut layers = Vec::new();
        for index in 0..summary.metadata.model.num_layers as usize {
            layers.push(LayerSpec {
                input_norm: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.input_layernorm.weight"),
                        &format!("model.layers.{index}.attention_norm.weight"),
                    ],
                )?,
                post_norm: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.post_attention_layernorm.weight"),
                        &format!("model.layers.{index}.ffn_norm.weight"),
                    ],
                )?,
                q_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.q_proj.weight"),
                        &format!("model.layers.{index}.attention.wq.weight"),
                    ],
                )?,
                q_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.q_proj.bias"),
                        &format!("model.layers.{index}.attention.wq.bias"),
                    ],
                ),
                k_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.k_proj.weight"),
                        &format!("model.layers.{index}.attention.wk.weight"),
                    ],
                )?,
                k_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.k_proj.bias"),
                        &format!("model.layers.{index}.attention.wk.bias"),
                    ],
                ),
                v_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.v_proj.weight"),
                        &format!("model.layers.{index}.attention.wv.weight"),
                    ],
                )?,
                v_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.v_proj.bias"),
                        &format!("model.layers.{index}.attention.wv.bias"),
                    ],
                ),
                o_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.o_proj.weight"),
                        &format!("model.layers.{index}.attention.wo.weight"),
                    ],
                )?,
                o_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.self_attn.o_proj.bias"),
                        &format!("model.layers.{index}.attention.wo.bias"),
                    ],
                ),
                gate_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.mlp.gate_proj.weight"),
                        &format!("model.layers.{index}.feed_forward.w1.weight"),
                    ],
                )?,
                gate_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.mlp.gate_proj.bias"),
                        &format!("model.layers.{index}.feed_forward.w1.bias"),
                    ],
                ),
                up_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.mlp.up_proj.weight"),
                        &format!("model.layers.{index}.feed_forward.w3.weight"),
                    ],
                )?,
                up_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.mlp.up_proj.bias"),
                        &format!("model.layers.{index}.feed_forward.w3.bias"),
                    ],
                ),
                down_proj: find_required_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.mlp.down_proj.weight"),
                        &format!("model.layers.{index}.feed_forward.w2.weight"),
                    ],
                )?,
                down_bias: find_optional_tensor(
                    tensors,
                    &[
                        &format!("model.layers.{index}.mlp.down_proj.bias"),
                        &format!("model.layers.{index}.feed_forward.w2.bias"),
                    ],
                ),
            });
        }

        Ok(Self {
            embedding,
            final_norm,
            lm_head,
            layers,
        })
    }
}

fn find_required_tensor(
    tensors: &std::collections::BTreeMap<String, TensorDescriptor>,
    candidates: &[&str],
) -> Result<String> {
    find_optional_tensor(tensors, candidates)
        .ok_or_else(|| anyhow!("missing required tensor; looked for {}", candidates.join(", ")))
}

fn find_optional_tensor(
    tensors: &std::collections::BTreeMap<String, TensorDescriptor>,
    candidates: &[&str],
) -> Option<String> {
    candidates
        .iter()
        .find_map(|candidate| tensors.contains_key(*candidate).then(|| (*candidate).to_string()))
}

struct RuntimeModel {
    bundle: LoadedBundle,
    vector_cache: HashMap<String, Vec<f32>>,
    codebook_cache: HashMap<String, Vec<f32>>,
    lora_cache: HashMap<String, LoraWeights>,
    kv_cache: Vec<LayerCache>,
}

#[derive(Default)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl RuntimeModel {
    fn new(bundle: LoadedBundle) -> Result<Self> {
        let layer_count = bundle.decoder.layers.len();
        Ok(Self {
            bundle,
            vector_cache: HashMap::new(),
            codebook_cache: HashMap::new(),
            lora_cache: HashMap::new(),
            kv_cache: (0..layer_count).map(|_| LayerCache::default()).collect(),
        })
    }

    fn load_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_manifest = self
            .bundle
            .summary
            .manifest
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("bundle does not declare tokenizer assets"))?;
        let tokenizer_file = tokenizer_manifest
            .files
            .iter()
            .find(|file| file.ends_with("tokenizer.json"))
            .ok_or_else(|| anyhow!("bundle does not include tokenizer.json; axonal currently requires a Hugging Face tokenizer.json bundle asset"))?;
        Tokenizer::from_file(self.bundle.bundle_dir.join(tokenizer_file))
            .map_err(|error| anyhow!("failed to load tokenizer: {error}"))
    }

    fn generate_ids_with_plan(&mut self, input_ids: &[u32], plan: &GenerationPlan) -> Result<Vec<u32>> {
        if input_ids.is_empty() {
            bail!("input_ids must not be empty");
        }
        let mut position = 0_usize;
        let mut logits = Vec::new();
        for &token in input_ids {
            logits = self.forward_step(token as usize, position)?;
            position += 1;
        }
        let mut generated = Vec::new();
        let eos = self.bundle.summary.metadata.runtime.eos_token_id.map(|id| id as usize);
        let mut sampler = plan.logits_processor();
        for _ in 0..plan.max_new_tokens {
            let next = sample_logits(&mut sampler, &logits)? as usize;
            generated.push(next as u32);
            if eos == Some(next) {
                break;
            }
            logits = self.forward_step(next, position)?;
            position += 1;
        }
        Ok(generated)
    }

    fn forward_step(&mut self, token_id: usize, position: usize) -> Result<Vec<f32>> {
        let embedding_name = self.bundle.decoder.embedding.clone();
        let mut hidden = self.embedding_row(&embedding_name, token_id)?;
        for layer_index in 0..self.bundle.decoder.layers.len() {
            hidden = self.forward_layer(layer_index, hidden, position)?;
        }
        let final_norm_name = self.bundle.decoder.final_norm.clone();
        let normalized = self.rms_norm(&hidden, &final_norm_name)?;
        let lm_head_name = self.bundle.decoder.lm_head.clone();
        self.linear(&lm_head_name, None, &normalized)
    }

    fn forward_layer(&mut self, layer_index: usize, mut hidden: Vec<f32>, position: usize) -> Result<Vec<f32>> {
        let layer = self.bundle.decoder.layers[layer_index].clone();
        let hidden_for_attn = self.rms_norm(&hidden, &layer.input_norm)?;
        let q = self.linear(&layer.q_proj, layer.q_bias.as_deref(), &hidden_for_attn)?;
        let k = self.linear(&layer.k_proj, layer.k_bias.as_deref(), &hidden_for_attn)?;
        let v = self.linear(&layer.v_proj, layer.v_bias.as_deref(), &hidden_for_attn)?;
        let q = self.apply_rope(q, self.bundle.summary.metadata.model.num_attention_heads as usize, position)?;
        let k = self.apply_rope(k, self.bundle.summary.metadata.model.num_kv_heads as usize, position)?;
        let attn = self.attention(layer_index, &q, &k, &v)?;
        let attn_proj = self.linear(&layer.o_proj, layer.o_bias.as_deref(), &attn)?;
        for (value, update) in hidden.iter_mut().zip(attn_proj) {
            *value += update;
        }

        let hidden_for_mlp = self.rms_norm(&hidden, &layer.post_norm)?;
        let gate = self.linear(&layer.gate_proj, layer.gate_bias.as_deref(), &hidden_for_mlp)?;
        let up = self.linear(&layer.up_proj, layer.up_bias.as_deref(), &hidden_for_mlp)?;
        let activated: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(gate_value, up_value)| silu(*gate_value) * *up_value)
            .collect();
        let down = self.linear(&layer.down_proj, layer.down_bias.as_deref(), &activated)?;
        for (value, update) in hidden.iter_mut().zip(down) {
            *value += update;
        }
        Ok(hidden)
    }

    fn rms_norm(&mut self, hidden: &[f32], weight_name: &str) -> Result<Vec<f32>> {
        let weights = self.vector(weight_name)?.clone();
        if weights.len() != hidden.len() {
            bail!("rms_norm weight shape mismatch for {weight_name}");
        }
        let mean_square = hidden.iter().map(|value| value * value).sum::<f32>() / hidden.len() as f32;
        let inv = 1.0 / (mean_square + 1e-6).sqrt();
        Ok(hidden
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| value * inv * weight)
            .collect())
    }

    fn linear(&mut self, weight_name: &str, bias_name: Option<&str>, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.bundle.tensor(weight_name)?.clone();
        let mut output = match tensor.dtype.as_str() {
            "fp16" | "bf16" | "fp32" => self.dense_matvec(weight_name, input)?,
            "axon_mxq" | "axon_nf2" | "axon_nf3" => self.quantized_matvec(weight_name, input)?,
            "axon_vq" => self.vq_matvec(weight_name, input)?,
            other => bail!("unsupported weight dtype {other} for linear"),
        };
        self.apply_lora_delta(weight_name, input, &mut output)?;
        if let Some(bias_name) = bias_name {
            let bias = self.vector(bias_name)?.clone();
            if bias.len() != output.len() {
                bail!("bias shape mismatch for {bias_name}");
            }
            for (value, offset) in output.iter_mut().zip(bias) {
                *value += offset;
            }
        }
        Ok(output)
    }

    fn apply_lora_delta(&mut self, weight_name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.ensure_lora_weights(weight_name)?;
        let Some(weights) = self.lora_cache.get(weight_name) else {
            return Ok(());
        };
        if weights.a_cols != input.len() {
            bail!(
                "LoRA A shape mismatch for {weight_name}: expected input width {}, got {}",
                weights.a_cols,
                input.len()
            );
        }
        if weights.b_rows != output.len() {
            bail!(
                "LoRA B shape mismatch for {weight_name}: expected output height {}, got {}",
                weights.b_rows,
                output.len()
            );
        }
        let mut projected = vec![0.0_f32; weights.a_rows];
        for row in 0..weights.a_rows {
            let start = row * weights.a_cols;
            let end = start + weights.a_cols;
            projected[row] = dot(&weights.a[start..end], input);
        }
        for row in 0..weights.b_rows {
            let start = row * weights.b_cols;
            let end = start + weights.b_cols;
            output[row] += dot(&weights.b[start..end], &projected) * weights.scale;
        }
        Ok(())
    }

    fn ensure_lora_weights(&mut self, weight_name: &str) -> Result<()> {
        if self.lora_cache.contains_key(weight_name) || self.bundle.lora_target(weight_name).is_none() {
            return Ok(());
        }
        let target = self
            .bundle
            .lora_target(weight_name)
            .cloned()
            .ok_or_else(|| anyhow!("missing LoRA target for {weight_name}"))?;
        let (a, a_rows, a_cols) = self.bundle.decode_lora_matrix(&target.a_name)?;
        let (b, b_rows, b_cols) = self.bundle.decode_lora_matrix(&target.b_name)?;
        if b_cols != a_rows {
            bail!(
                "LoRA factors are incompatible for {weight_name}: A is {}x{}, B is {}x{}",
                a_rows,
                a_cols,
                b_rows,
                b_cols
            );
        }
        self.lora_cache.insert(
            weight_name.to_string(),
            LoraWeights {
                a,
                a_rows,
                a_cols,
                b,
                b_rows,
                b_cols,
                scale: target.scale,
            },
        );
        Ok(())
    }

    fn dense_matvec(&self, weight_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.bundle.tensor(weight_name)?;
        let (rows, cols) = matrix_shape(tensor)?;
        if cols != input.len() {
            bail!("input size mismatch for {weight_name}: expected {cols}, got {}", input.len());
        }
        let payload = self.bundle.tensor_bytes(tensor)?;
        let values = match tensor.dtype.as_str() {
            "fp16" => decode_f16_bytes(payload),
            "bf16" => decode_bf16_bytes(payload),
            "fp32" => decode_f32_bytes(payload),
            other => bail!("unsupported dense dtype {other}"),
        };
        Ok(dense_matvec_rows(&values, rows, cols, input))
    }

    fn quantized_matvec(&self, weight_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.bundle.tensor(weight_name)?;
        let (rows, cols) = matrix_shape(tensor)?;
        if cols != input.len() {
            bail!("input size mismatch for {weight_name}: expected {cols}, got {}", input.len());
        }
        let bits = match tensor.dtype.as_str() {
            "axon_mxq" => tensor.bits.ok_or_else(|| anyhow!("MXQ tensor missing bits"))? as usize,
            "axon_nf2" => 2,
            "axon_nf3" => 3,
            other => bail!("unsupported quantized dtype {other}"),
        };
        let group_size = tensor.group_size.ok_or_else(|| anyhow!("MXQ tensor missing group_size"))? as usize;
        let payload = self.bundle.tensor_bytes(tensor)?;
        let packed_group_bytes = (group_size * bits).div_ceil(8);
        let group_bytes = 2 + packed_group_bytes;
        let mut output = vec![0.0_f32; rows];
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let mut flat_index = row_start;
            let mut input_index = 0;
            while flat_index < row_end {
                let group_index = flat_index / group_size;
                let within_group = flat_index % group_size;
                let available = group_size - within_group;
                let take = available.min(row_end - flat_index);
                let group_offset = group_index * group_bytes;
                let scale_bits = u16::from_le_bytes([payload[group_offset], payload[group_offset + 1]]);
                let scale = f16::from_bits(scale_bits).to_f32();
                let packed = &payload[group_offset + 2..group_offset + group_bytes];
                let codes = unpack_codes(packed, bits, group_size);
                for index in 0..take {
                    let value = if tensor.dtype == "axon_mxq" {
                        let zero_point = 1_i32 << (bits - 1);
                        ((codes[within_group + index] as i32 - zero_point) as f32) * scale
                    } else {
                        scale * nf_code_to_value(bits, codes[within_group + index])?
                    };
                    output[row] += value * input[input_index + index];
                }
                flat_index += take;
                input_index += take;
            }
        }
        if let Some(outlier_bytes) = self.bundle.outlier_bytes(tensor)? {
            apply_outlier_dot(tensor, outlier_bytes, input, &mut output)?;
        }
        Ok(output)
    }

    fn vq_matvec(&mut self, weight_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.bundle.tensor(weight_name)?.clone();
        let (rows, cols) = matrix_shape(&tensor)?;
        if cols != input.len() {
            bail!("input size mismatch for {weight_name}: expected {cols}, got {}", input.len());
        }
        let vq_dim = tensor.vq_dim.ok_or_else(|| anyhow!("VQ tensor missing vq_dim"))? as usize;
        let codes = self.bundle.tensor_bytes(&tensor)?.to_vec();
        let codebook_id = tensor
            .codebook_id
            .clone()
            .ok_or_else(|| anyhow!("VQ tensor missing codebook_id"))?;
        let centers = self.codebook_centers(&codebook_id)?.clone();
        let segments = cols / vq_dim;
        let mut output = vec![0.0_f32; rows];
        for row in 0..rows {
            let row_codes = &codes[row * segments..(row + 1) * segments];
            output[row] = vq_row_dot(row_codes, &centers, vq_dim, input);
        }
        Ok(output)
    }

    fn embedding_row(&mut self, weight_name: &str, row: usize) -> Result<Vec<f32>> {
        let tensor = self.bundle.tensor(weight_name)?.clone();
        let (rows, cols) = matrix_shape(&tensor)?;
        if row >= rows {
            bail!("token id {row} is out of range for embedding {weight_name}");
        }
        let payload = self.bundle.tensor_bytes(&tensor)?.to_vec();
        let mut output = match tensor.dtype.as_str() {
            "fp16" => decode_dense_row_fp16(&payload, row, cols),
            "bf16" => decode_dense_row_bf16(&payload, row, cols),
            "fp32" => decode_dense_row_fp32(&payload, row, cols),
            "axon_mxq" | "axon_nf2" | "axon_nf3" => decode_quantized_row(&tensor, &payload, row, cols)?,
            "axon_vq" => {
                let vq_dim = tensor.vq_dim.ok_or_else(|| anyhow!("VQ tensor missing vq_dim"))? as usize;
                let codebook_id = tensor
                    .codebook_id
                    .clone()
                    .ok_or_else(|| anyhow!("VQ tensor missing codebook_id"))?;
                let centers = self.codebook_centers(&codebook_id)?.clone();
                let segments = cols / vq_dim;
                let row_codes = &payload[row * segments..(row + 1) * segments];
                vq_row_decode(row_codes, &centers, vq_dim)
            }
            other => bail!("unsupported embedding dtype {other}"),
        };
        if let Some(outlier_bytes) = self.bundle.outlier_bytes(&tensor)? {
            apply_outlier_row(&tensor, outlier_bytes, row, &mut output)?;
        }
        Ok(output)
    }

    fn attention(&mut self, layer_index: usize, q: &[f32], k: &[f32], v: &[f32]) -> Result<Vec<f32>> {
        let heads = self.bundle.summary.metadata.model.num_attention_heads as usize;
        let kv_heads = self.bundle.summary.metadata.model.num_kv_heads as usize;
        let head_dim = self.bundle.summary.metadata.model.head_dim as usize;
        let group = (heads / kv_heads.max(1)).max(1);
        let cache = &mut self.kv_cache[layer_index];
        cache.keys.extend_from_slice(k);
        cache.values.extend_from_slice(v);
        let seq_len = cache.keys.len() / (kv_heads * head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0_f32; heads * head_dim];

        for head in 0..heads {
            let q_slice = &q[head * head_dim..(head + 1) * head_dim];
            let kv_head = head / group;
            let mut scores = vec![0.0_f32; seq_len];
            for (position, score) in scores.iter_mut().enumerate() {
                let offset = (position * kv_heads + kv_head) * head_dim;
                let k_slice = &cache.keys[offset..offset + head_dim];
                *score = dot(q_slice, k_slice) * scale;
            }
            softmax(&mut scores);
            let out_slice = &mut output[head * head_dim..(head + 1) * head_dim];
            for (position, score) in scores.iter().enumerate() {
                let offset = (position * kv_heads + kv_head) * head_dim;
                let v_slice = &cache.values[offset..offset + head_dim];
                for index in 0..head_dim {
                    out_slice[index] += *score * v_slice[index];
                }
            }
        }
        Ok(output)
    }

    fn apply_rope(&self, mut values: Vec<f32>, heads: usize, position: usize) -> Result<Vec<f32>> {
        let head_dim = self.bundle.summary.metadata.model.head_dim as usize;
        let theta = self
            .bundle
            .summary
            .metadata
            .model
            .rope
            .theta
            .unwrap_or(10_000.0) as f32;
        if values.len() != heads * head_dim {
            bail!("RoPE shape mismatch");
        }
        for head in 0..heads {
            apply_rope_head(
                &mut values[head * head_dim..(head + 1) * head_dim],
                position,
                theta,
            );
        }
        Ok(values)
    }

    fn vector(&mut self, name: &str) -> Result<&Vec<f32>> {
        if !self.vector_cache.contains_key(name) {
            let tensor = self.bundle.tensor(name)?.clone();
            let payload = self.bundle.tensor_bytes(&tensor)?;
            let values = match tensor.dtype.as_str() {
                "fp16" => decode_f16_bytes(payload),
                "bf16" => decode_bf16_bytes(payload),
                "fp32" => decode_f32_bytes(payload),
                other => bail!("unsupported vector dtype {other} for {name}"),
            };
            self.vector_cache.insert(name.to_string(), values);
        }
        self.vector_cache
            .get(name)
            .ok_or_else(|| anyhow!("missing cached vector {name}"))
    }

    fn codebook_centers(&mut self, codebook_id: &str) -> Result<&Vec<f32>> {
        if !self.codebook_cache.contains_key(codebook_id) {
            let centers = decode_f16_bytes(self.bundle.codebook_bytes(codebook_id)?);
            self.codebook_cache.insert(codebook_id.to_string(), centers);
        }
        self.codebook_cache
            .get(codebook_id)
            .ok_or_else(|| anyhow!("missing cached codebook {codebook_id}"))
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

fn matrix_shape(tensor: &TensorDescriptor) -> Result<(usize, usize)> {
    if tensor.shape.len() < 2 {
        bail!("tensor is not a matrix");
    }
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1..].iter().product::<u64>() as usize;
    Ok((rows, cols))
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

fn decode_dense_row_fp16(bytes: &[u8], row: usize, cols: usize) -> Vec<f32> {
    let start = row * cols * 2;
    decode_f16_bytes(&bytes[start..start + cols * 2])
}

fn decode_dense_row_bf16(bytes: &[u8], row: usize, cols: usize) -> Vec<f32> {
    let start = row * cols * 2;
    decode_bf16_bytes(&bytes[start..start + cols * 2])
}

fn decode_dense_row_fp32(bytes: &[u8], row: usize, cols: usize) -> Vec<f32> {
    let start = row * cols * 4;
    decode_f32_bytes(&bytes[start..start + cols * 4])
}

fn dense_matvec_rows(values: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows];
    for row in 0..rows {
        output[row] = dot(&values[row * cols..(row + 1) * cols], input);
    }
    output
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

fn decode_quantized_row(tensor: &TensorDescriptor, payload: &[u8], row: usize, cols: usize) -> Result<Vec<f32>> {
    let bits = match tensor.dtype.as_str() {
        "axon_mxq" => tensor.bits.ok_or_else(|| anyhow!("MXQ tensor missing bits"))? as usize,
        "axon_nf2" => 2,
        "axon_nf3" => 3,
        other => bail!("unsupported quantized dtype {other}"),
    };
    let group_size = tensor.group_size.ok_or_else(|| anyhow!("quantized tensor missing group_size"))? as usize;
    let packed_group_bytes = (group_size * bits).div_ceil(8);
    let group_bytes = 2 + packed_group_bytes;
    let row_start = row * cols;
    let row_end = row_start + cols;
    let mut output = vec![0.0_f32; cols];
    let mut flat_index = row_start;
    let mut output_index = 0;
    while flat_index < row_end {
        let group_index = flat_index / group_size;
        let within_group = flat_index % group_size;
        let available = group_size - within_group;
        let take = available.min(row_end - flat_index);
        let group_offset = group_index * group_bytes;
        let scale_bits = u16::from_le_bytes([payload[group_offset], payload[group_offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let packed = &payload[group_offset + 2..group_offset + group_bytes];
        let codes = unpack_codes(packed, bits, group_size);
        for index in 0..take {
            output[output_index + index] = if tensor.dtype == "axon_mxq" {
                let zero_point = 1_i32 << (bits - 1);
                ((codes[within_group + index] as i32 - zero_point) as f32) * scale
            } else {
                scale * nf_code_to_value(bits, codes[within_group + index])?
            };
        }
        flat_index += take;
        output_index += take;
    }
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

fn vq_row_decode(codes: &[u8], centers: &[f32], vq_dim: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(codes.len() * vq_dim);
    for &code in codes {
        let center = &centers[(code as usize) * vq_dim..(code as usize + 1) * vq_dim];
        output.extend_from_slice(center);
    }
    output
}

fn vq_row_dot(codes: &[u8], centers: &[f32], vq_dim: usize, input: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (segment, &code) in codes.iter().enumerate() {
        let center = &centers[(code as usize) * vq_dim..(code as usize + 1) * vq_dim];
        let input_slice = &input[segment * vq_dim..(segment + 1) * vq_dim];
        sum += dot(center, input_slice);
    }
    sum
}

fn apply_outlier_dot(
    tensor: &TensorDescriptor,
    outlier_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let rows = tensor.shape[0] as usize;
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
    let values = decode_f16_bytes(&outlier_bytes[value_offset..]);
    for row in 0..rows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for index in start..end {
            let col = col_idx[index] as usize;
            if col < input.len() {
                output[row] += values[index] * input[col];
            }
        }
    }
    Ok(())
}

fn apply_outlier_row(
    tensor: &TensorDescriptor,
    outlier_bytes: &[u8],
    row: usize,
    output: &mut [f32],
) -> Result<()> {
    let rows = tensor.shape[0] as usize;
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
    let values = decode_f16_bytes(&outlier_bytes[value_offset..]);
    let start = row_ptr[row] as usize;
    let end = row_ptr[row + 1] as usize;
    for index in start..end {
        let col = col_idx[index] as usize;
        if col < output.len() {
            output[col] += values[index];
        }
    }
    Ok(())
}

fn apply_rope_head(values: &mut [f32], position: usize, theta: f32) {
    let head_dim = values.len();
    for index in (0..head_dim).step_by(2) {
        if index + 1 >= head_dim {
            break;
        }
        let inv_freq = 1.0 / theta.powf(index as f32 / head_dim as f32);
        let angle = position as f32 * inv_freq;
        let cos = angle.cos();
        let sin = angle.sin();
        let first = values[index];
        let second = values[index + 1];
        values[index] = first * cos - second * sin;
        values[index + 1] = first * sin + second * cos;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn softmax(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max_value = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - max_value).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sample_logits(processor: &mut LogitsProcessor, logits: &[f32]) -> Result<u32> {
    let tensor = Tensor::from_slice(logits, logits.len(), &Device::Cpu)?;
    processor.sample(&tensor).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FLAG_HAS_CHECKSUMS, FLAG_HAS_LORA_DELTA};
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;
    use xxhash_rust::xxh64::xxh64;

    fn align64(value: u64) -> u64 {
        if value % 64 == 0 {
            value
        } else {
            value + (64 - (value % 64))
        }
    }

    fn summary_with_tokenizer(files: Vec<&str>) -> BundleSummary {
        serde_json::from_value(serde_json::json!({
            "manifest": {
                "manifest_version": "2.0.0-draft",
                "task": "causal_lm",
                "model_file": "model.axon",
                "config_file": null,
                "generation_config_file": null,
                "tokenizer": {
                    "kind": "huggingface",
                    "files": files
                }
            },
            "metadata": {
                "format": "axon",
                "version": "2.0.0-draft",
                "task": "causal_lm",
                "architecture": "LlamaForCausalLM",
                "model_family": "tiny-test",
                "model": {
                    "hidden_dim": 4,
                    "intermediate_dim": 4,
                    "num_layers": 1,
                    "num_attention_heads": 1,
                    "num_kv_heads": 1,
                    "head_dim": 4,
                    "context_length": 32,
                    "vocab_size": 4,
                    "rope": {"type": "rope", "theta": 10000.0, "scaling": null},
                    "total_parameter_count": null,
                    "active_parameter_count": null,
                    "moe": null
                },
                "runtime": {
                    "bos_token_id": 0,
                    "eos_token_id": 3,
                    "pad_token_id": null,
                    "default_generation": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "max_new_tokens": 1
                    }
                },
                "source": {
                    "format": "hf_safetensors",
                    "identifier": "/tmp/tiny",
                    "conversion_tool": "test",
                    "conversion_time_utc": "2026-03-21T00:00:00Z"
                },
                "avg_bits_per_weight": 32.0,
                "quant_method": "none",
                "calibration": null,
                "tensors": {},
                "codebooks": {},
                "hw_hints": {}
            },
            "file_size": 0,
            "flags": 0
        }))
        .unwrap()
    }

    fn descriptor(shape: &[u64], offset: u64, bytes: u64, stream_order: u32, checksum: String) -> serde_json::Value {
        serde_json::json!({
            "shape": shape,
            "dtype": "fp32",
            "bits": null,
            "group_size": null,
            "source_tensor_name": null,
            "data_offset": offset,
            "data_bytes": bytes,
            "scale_interleaved": false,
            "outlier_indices_offset": null,
            "outlier_count": null,
            "sensitivity_score": null,
            "stream_order": stream_order,
            "codebook_id": null,
            "vq_dim": null,
            "checksum_xxh64": checksum
        })
    }

    fn write_tiny_decoder_bundle(dir: &Path) {
        let mut tensors: Vec<(&str, Vec<f32>, Vec<u64>)> = vec![
            ("model.embed_tokens.weight", vec![1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0], vec![4,4]),
            ("model.layers.0.input_layernorm.weight", vec![1.0,1.0,1.0,1.0], vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![1.0,1.0,1.0,1.0], vec![4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.layers.0.mlp.up_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.layers.0.mlp.down_proj.weight", vec![0.0;16], vec![4,4]),
            ("model.norm.weight", vec![1.0,1.0,1.0,1.0], vec![4]),
            ("lm_head.weight", vec![1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0], vec![4,4]),
        ];

        let mut data_region = Vec::new();
        let mut tensor_json = serde_json::Map::new();
        let mut relative_offset = 0_u64;
        for (stream_order, (name, values, shape)) in tensors.drain(..).enumerate() {
            relative_offset = align64(relative_offset);
            if data_region.len() < relative_offset as usize {
                data_region.resize(relative_offset as usize, 0);
            }
            let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
            let checksum = format!("{:016x}", xxh64(&bytes, 0));
            tensor_json.insert(
                name.to_string(),
                descriptor(&shape, relative_offset, bytes.len() as u64, stream_order as u32, checksum),
            );
            data_region.extend_from_slice(&bytes);
            relative_offset += bytes.len() as u64;
        }

        let metadata = serde_json::json!({
            "format": "axon",
            "version": "2.0.0-draft",
            "task": "causal_lm",
            "architecture": "LlamaForCausalLM",
            "model_family": "tiny-test",
            "model": {
                "hidden_dim": 4,
                "intermediate_dim": 4,
                "num_layers": 1,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 4,
                "context_length": 32,
                "vocab_size": 4,
                "rope": {"type": "rope", "theta": 10000.0, "scaling": null}
            },
            "runtime": {
                "bos_token_id": 0,
                "eos_token_id": 3,
                "pad_token_id": null,
                "default_generation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 1
                }
            },
            "source": {
                "format": "hf_safetensors",
                "identifier": "/tmp/tiny",
                "conversion_tool": "test",
                "conversion_time_utc": "2026-03-21T00:00:00Z"
            },
            "avg_bits_per_weight": 32.0,
            "quant_method": "none",
            "calibration": null,
            "tensors": tensor_json,
            "codebooks": {},
            "hw_hints": {}
        });

        let metadata_bytes = serde_json::to_vec(&metadata).unwrap();
        let data_offset = align64(64 + metadata_bytes.len() as u64);
        let mut header = [0_u8; 64];
        header[0..4].copy_from_slice(b"AXON");
        header[4..6].copy_from_slice(&2_u16.to_le_bytes());
        header[6..8].copy_from_slice(&0_u16.to_le_bytes());
        header[8..12].copy_from_slice(&(FLAG_HAS_CHECKSUMS | (1 << 2)).to_le_bytes());
        header[12..16].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[16..24].copy_from_slice(&data_offset.to_le_bytes());

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
        file.write_all(&vec![0_u8; (data_offset - (64 + metadata_bytes.len() as u64)) as usize])
            .unwrap();
        file.write_all(&data_region).unwrap();
    }

    fn write_tiny_lora_bundle(dir: &Path, base_model: &str) {
        let a = vec![0.0_f32, 0.0, 1.0, 0.0];
        let b = vec![0.0_f32, 4.0, 0.0, 0.0];
        let a_bytes: Vec<u8> = a
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = b
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();

        let mut lora_region = Vec::new();
        let mut tensor_json = serde_json::Map::new();
        let mut relative_offset = 0_u64;
        for (stream_order, (name, bytes, shape)) in [
            ("lm_head.weight.lora_A", a_bytes, vec![1_u64, 4_u64]),
            ("lm_head.weight.lora_B", b_bytes, vec![4_u64, 1_u64]),
        ]
        .into_iter()
        .enumerate()
        {
            relative_offset = align64(relative_offset);
            if lora_region.len() < relative_offset as usize {
                lora_region.resize(relative_offset as usize, 0);
            }
            tensor_json.insert(
                name.to_string(),
                serde_json::json!({
                    "shape": shape,
                    "dtype": "fp16",
                    "bits": null,
                    "group_size": null,
                    "source_tensor_name": null,
                    "data_offset": relative_offset,
                    "data_bytes": bytes.len(),
                    "scale_interleaved": false,
                    "outlier_indices_offset": null,
                    "outlier_count": null,
                    "sensitivity_score": null,
                    "stream_order": stream_order,
                    "per_head_bits": null,
                    "nf_scale_fp16": false,
                    "smoothquant_scale": null,
                    "prefetch_priority": 0.9,
                    "codebook_id": null,
                    "vq_dim": null,
                    "dedup_canonical": null,
                    "dedup_correction_offset": null,
                    "dedup_correction_count": null,
                    "lora_rank": 1,
                    "lora_alpha": 1,
                    "target": "lm_head.weight",
                    "checksum_xxh64": null
                }),
            );
            lora_region.extend_from_slice(&bytes);
            relative_offset += bytes.len() as u64;
        }

        let mut region_offset = 0_u64;
        let metadata_bytes = loop {
            let metadata = serde_json::json!({
                "format": "axon",
                "version": "2.0.0-draft",
                "task": "causal_lm",
                "architecture": "LlamaForCausalLM",
                "model_family": "tiny-test-lora",
                "model": {
                    "hidden_dim": 4,
                    "intermediate_dim": 4,
                    "num_layers": 1,
                    "num_attention_heads": 1,
                    "num_kv_heads": 1,
                    "head_dim": 4,
                    "context_length": 32,
                    "vocab_size": 4,
                    "rope": {"type": "rope", "theta": 10000.0, "scaling": null}
                },
                "runtime": {
                    "bos_token_id": 0,
                    "eos_token_id": 3,
                    "pad_token_id": null,
                    "default_generation": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "max_new_tokens": 1
                    }
                },
                "source": {
                    "format": "hf_safetensors",
                    "identifier": "/tmp/tiny-lora",
                    "conversion_tool": "test",
                    "conversion_time_utc": "2026-03-21T00:00:00Z"
                },
                "avg_bits_per_weight": 16.0,
                "quant_method": "lora-delta",
                "calibration": null,
                "tensors": tensor_json,
                "codebooks": {},
                "hw_hints": {},
                "lora": {
                    "base_model": base_model,
                    "base_hash": "ignored-in-test",
                    "rank": 1,
                    "alpha": 1,
                    "target_modules": ["lm_head"],
                    "region_offset": region_offset,
                    "region_bytes": lora_region.len()
                }
            });
            let bytes = serde_json::to_vec(&metadata).unwrap();
            let next_offset = align64(64 + bytes.len() as u64);
            if next_offset == region_offset {
                break bytes;
            }
            region_offset = next_offset;
        };

        let mut header = [0_u8; 64];
        header[0..4].copy_from_slice(b"AXON");
        header[4..6].copy_from_slice(&2_u16.to_le_bytes());
        header[6..8].copy_from_slice(&0_u16.to_le_bytes());
        header[8..12].copy_from_slice(&FLAG_HAS_LORA_DELTA.to_le_bytes());
        header[12..16].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[16..24].copy_from_slice(&region_offset.to_le_bytes());
        header[24..32].copy_from_slice(&region_offset.to_le_bytes());
        header[32..40].copy_from_slice(&region_offset.to_le_bytes());

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
        file.write_all(&vec![0_u8; (region_offset - (64 + metadata_bytes.len() as u64)) as usize])
            .unwrap();
        file.write_all(&lora_region).unwrap();
    }

    #[test]
    fn tiny_decoder_generates_same_token() {
        let temp = tempdir().unwrap();
        write_tiny_decoder_bundle(temp.path());
        let generated = generate_token_ids(temp.path(), &[2], 1).unwrap();
        assert_eq!(generated, vec![2]);
    }

    #[test]
    fn tiny_decoder_applies_lora_delta_bundle() {
        let temp = tempdir().unwrap();
        let base_bundle = temp.path().join("tiny-base");
        let lora_bundle = temp.path().join("tiny-lora");
        write_tiny_decoder_bundle(&base_bundle);
        write_tiny_lora_bundle(&lora_bundle, "tiny-base.axon");
        let generated = generate_token_ids(&lora_bundle, &[2], 1).unwrap();
        assert_eq!(generated, vec![1]);
    }

    #[test]
    fn render_prompt_uses_chat_template_file_fallback() {
        let temp = tempdir().unwrap();
        fs::write(
            temp.path().join("chat_template.jinja"),
            "{{ messages[0].content }}<|assistant|>",
        )
        .unwrap();
        let summary = summary_with_tokenizer(vec!["chat_template.jinja"]);
        let (rendered, raw, note) = render_prompt(&summary, temp.path(), "Hello", false).unwrap();
        assert_eq!(rendered, "Hello<|assistant|>");
        assert!(!raw);
        assert_eq!(note.as_deref(), Some("applied chat template"));
    }
}
