use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result, anyhow, bail};
use axonal::{
    GenerationPreview, RegisteredModel, RunOptions, ShowResponse, get_registered_model, load_bundle,
    run_model_with_options, scan_registry, show_bundle, tags_from_registry,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use rand::RngCore;
use reqwest::blocking::Response;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::blocking::Client;
use reqwest::Url;
use rpassword::prompt_password;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[derive(Parser)]
#[command(name = "axonal", about = "Local runtime surface for AXON bundles")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Inspect {
        #[arg(long)]
        bundle: PathBuf,
    },
    Convert {
        source: String,
        #[arg(long, help = "Treat SOURCE as a local directory or gguf file instead of a Hugging Face repo id")]
        local: bool,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, value_parser = ["hf", "gguf"])]
        source_format: Option<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        library: Option<PathBuf>,
        #[arg(long)]
        replace: bool,
        #[arg(long, value_parser = ["none", "auto", "mxq"], default_value = "auto")]
        quantization: String,
        #[arg(long)]
        group_size: Option<usize>,
        #[arg(long, default_value_t = 6.0)]
        outlier_sigma: f64,
        #[arg(long)]
        no_vq: bool,
    },
    Run {
        model: String,
        prompt: Vec<String>,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        max_tokens: Option<usize>,
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        raw: bool,
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        verbose: bool,
        #[arg(long)]
        nowordwrap: bool,
    },
    Show {
        model: String,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        host: Option<String>,
        #[arg(short = 'v', long)]
        verbose: bool,
    },
    #[command(alias = "ls")]
    List {
        #[arg(long)]
        host: Option<String>,
    },
    Ps {
        #[arg(long)]
        host: Option<String>,
        #[arg(short = 'v', long)]
        verbose: bool,
    },
    Serve {
        #[arg(long)]
        models: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 11435)]
        port: u16,
    },
    Hf {
        #[command(subcommand)]
        command: HuggingFaceCommand,
    },
}

#[derive(Subcommand)]
enum HuggingFaceCommand {
    Login {
        #[arg(long)]
        token: Option<String>,
        #[arg(long, help = "Seal the token with the system TPM instead of using the default local encrypted store")]
        tpm: bool,
    },
    Logout,
    Status,
}

#[derive(Clone)]
struct AppState {
    models: Arc<Vec<RegisteredModel>>,
}

#[derive(Debug, Deserialize)]
struct ShowRequest {
    name: Option<String>,
    path: Option<String>,
    backend: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    model: String,
    path: Option<String>,
    prompt: String,
    backend: Option<String>,
    num_predict: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: Option<u64>,
    raw: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TagsResponse {
    models: Vec<axonal::ModelTag>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PsResponse {
    models: Vec<axonal::RuntimeProcessInfo>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceModelInfo {
    #[serde(default)]
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
}

struct TempWorkspace {
    path: PathBuf,
}

impl TempWorkspace {
    fn new(prefix: &str) -> Result<Self> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = env::temp_dir().join(format!("{prefix}-{}-{unique}", std::process::id()));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create temporary workspace {}", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { bundle } => {
            let summary = load_bundle(&bundle)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Convert {
            source,
            local,
            revision,
            source_format,
            name,
            library,
            replace,
            quantization,
            group_size,
            outlier_sigma,
            no_vq,
        } => {
            let installed = convert_command(
                &source,
                local,
                &revision,
                source_format.as_deref(),
                name.as_deref(),
                library.as_deref(),
                replace,
                &quantization,
                group_size,
                outlier_sigma,
                no_vq,
            )?;
            println!("{}", installed.display());
        }
        Command::Run {
            model,
            prompt,
            backend,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            raw,
            host,
            verbose,
            nowordwrap,
        } => {
            let prompt = collect_prompt(prompt)?;
            let options = RunOptions {
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                raw_prompt: raw,
            };
            let response = run_command(
                &model,
                &prompt,
                host.as_deref(),
                backend.as_deref(),
                options,
            )?;
            render_run_output(&response, verbose, nowordwrap)?;
        }
        Command::Show {
            model,
            backend,
            host,
            verbose,
        } => {
            let response = show_command(&model, host.as_deref(), backend.as_deref())?;
            if verbose {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                let params = format_parameter_summary(
                    response.summary.metadata.model.active_parameter_count,
                    response.summary.metadata.model.total_parameter_count,
                );
                println!(
                    "{}\nfamily: {}\nparams: {}\nbackend: {} ({})\nquantized tensors: {}",
                    response.summary.metadata.architecture,
                    response.summary.metadata.model_family,
                    params,
                    response.backend.backend,
                    response.backend.kernel,
                    response.stats.quantized_tensors
                );
            }
        }
        Command::List { host } => {
            let tags = list_command(host.as_deref())?;
            print_tags(&tags.models);
        }
        Command::Ps { host, verbose } => {
            let processes = ps_command(host.as_deref())?;
            if verbose {
                println!("{}", serde_json::to_string_pretty(&processes)?);
            } else {
                print_processes(&processes.models);
            }
        }
        Command::Serve { models, host, port } => {
            let model_root = models.unwrap_or_else(default_models_dir);
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async move {
                let registry = scan_registry(&model_root)?;
                let state = AppState {
                    models: Arc::new(registry),
                };
                let app = Router::new()
                    .route("/api/tags", get(tags))
                    .route("/api/ps", get(ps))
                    .route("/api/show", post(show))
                    .route("/api/generate", post(generate))
                    .with_state(state);
                let listener = TcpListener::bind(format!("{host}:{port}")).await?;
                axum::serve(listener, app).await?;
                Result::<()>::Ok(())
            })?;
        }
        Command::Hf { command } => match command {
            HuggingFaceCommand::Login { token, tpm } => {
                huggingface_login_command(token.as_deref(), tpm)?;
            }
            HuggingFaceCommand::Logout => {
                huggingface_logout_command()?;
            }
            HuggingFaceCommand::Status => {
                huggingface_status_command()?;
            }
        },
    }
    Ok(())
}

async fn tags(State(state): State<AppState>) -> Json<TagsResponse> {
    Json(TagsResponse {
        models: tags_from_registry(&state.models),
    })
}

async fn ps() -> Json<PsResponse> {
    Json(PsResponse {
        models: axonal::accelerated::cached_runtime_processes(),
    })
}

async fn show(State(state): State<AppState>, Json(request): Json<ShowRequest>) -> impl IntoResponse {
    match lookup_show(&state.models, request.name.as_deref(), request.path.as_deref(), request.backend.as_deref()) {
        Ok(response) => (StatusCode::OK, Json(json!(response))).into_response(),
        Err(error) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": error.to_string() })),
        )
            .into_response(),
    }
}

async fn generate(State(state): State<AppState>, Json(request): Json<GenerateRequest>) -> impl IntoResponse {
    match resolve_generate_bundle(&state.models, request.path.as_deref(), &request.model)
        .and_then(|bundle| {
            run_model_with_options(
                &bundle,
                &request.prompt,
                request.backend.as_deref(),
                RunOptions {
                    max_tokens: request.num_predict,
                    temperature: request.temperature,
                    top_p: request.top_p,
                    top_k: request.top_k,
                    seed: request.seed,
                    raw_prompt: request.raw.unwrap_or(false),
                },
            )
        })
    {
        Ok(response) => (StatusCode::OK, Json(json!(response))).into_response(),
        Err(error) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": error.to_string() })),
        )
            .into_response(),
    }
}

fn collect_prompt(parts: Vec<String>) -> Result<String> {
    if !parts.is_empty() {
        return Ok(parts.join(" "));
    }
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer.trim().to_string())
}

fn axonal_state_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        return home.join(".axonal");
    }
    PathBuf::from(".axonal")
}

fn huggingface_credential_dir() -> PathBuf {
    axonal_state_dir().join("credentials").join("huggingface")
}

fn huggingface_encrypted_token_path() -> PathBuf {
    huggingface_credential_dir().join("token.enc")
}

fn huggingface_public_blob_path() -> PathBuf {
    huggingface_credential_dir().join("token.pub")
}

fn huggingface_private_blob_path() -> PathBuf {
    huggingface_credential_dir().join("token.priv")
}

fn huggingface_token_from_env() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("HUGGING_FACE_HUB_TOKEN")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

fn huggingface_local_store_exists() -> bool {
    huggingface_encrypted_token_path().is_file()
}

fn huggingface_tpm_store_exists() -> bool {
    huggingface_public_blob_path().is_file() && huggingface_private_blob_path().is_file()
}

fn huggingface_store_exists() -> bool {
    huggingface_local_store_exists() || huggingface_tpm_store_exists()
}

fn huggingface_auth_token() -> Result<Option<String>> {
    if let Some(token) = huggingface_token_from_env() {
        return Ok(Some(token));
    }
    if let Some(token) = load_huggingface_token_from_local_store()? {
        return Ok(Some(token));
    }
    load_huggingface_token_from_tpm_store()
}

fn huggingface_login_command(token_arg: Option<&str>, use_tpm: bool) -> Result<()> {
    let token = if let Some(token) = token_arg {
        token.trim().to_string()
    } else {
        if !io::stdin().is_terminal() {
            bail!("cannot prompt for a Hugging Face token without a TTY; pass --token or set HF_TOKEN");
        }
        prompt_password("Hugging Face token: ")
            .context("failed to read Hugging Face token")?
            .trim()
            .to_string()
    };
    if token.is_empty() {
        bail!("Hugging Face token cannot be empty");
    }
    if use_tpm {
        store_huggingface_token_in_tpm(&token)?;
        println!(
            "Stored Hugging Face token in TPM-backed credential store at {}",
            huggingface_credential_dir().display()
        );
    } else {
        store_huggingface_token_in_local_store(&token)?;
        println!(
            "Stored Hugging Face token in local encrypted store at {}",
            huggingface_encrypted_token_path().display()
        );
    }
    Ok(())
}

fn huggingface_logout_command() -> Result<()> {
    if !huggingface_store_exists() {
        println!("No stored Hugging Face token.");
        return Ok(());
    }
    if huggingface_local_store_exists() {
        fs::remove_file(huggingface_encrypted_token_path())
            .with_context(|| "failed to remove stored Hugging Face encrypted token")?;
    }
    if huggingface_tpm_store_exists() {
        fs::remove_file(huggingface_public_blob_path())
            .with_context(|| "failed to remove stored Hugging Face public blob")?;
        fs::remove_file(huggingface_private_blob_path())
            .with_context(|| "failed to remove stored Hugging Face private blob")?;
    }
    println!("Removed stored Hugging Face token.");
    Ok(())
}

fn huggingface_status_command() -> Result<()> {
    let mut found = false;
    if huggingface_token_from_env().is_some() {
        println!("Hugging Face auth: environment variable");
        found = true;
    }
    if huggingface_local_store_exists() {
        println!(
            "Hugging Face auth: local encrypted store at {}",
            huggingface_encrypted_token_path().display()
        );
        found = true;
    }
    if huggingface_tpm_store_exists() {
        println!(
            "Hugging Face auth: TPM-backed credential store at {}",
            huggingface_credential_dir().display()
        );
        found = true;
    }
    if !found {
        println!("Hugging Face auth: not configured");
    }
    Ok(())
}

fn maybe_prompt_for_huggingface_login() -> Result<()> {
    if huggingface_token_from_env().is_some() || huggingface_store_exists() {
        return Ok(());
    }
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Ok(());
    }
    eprint!(
        "[axonal] no Hugging Face login is configured. Store one in the local encrypted store now? [y/N] (use `axonal hf login --tpm` for TPM-backed storage): "
    );
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("failed to read Hugging Face login prompt")?;
    let answer = answer.trim().to_ascii_lowercase();
    if answer == "y" || answer == "yes" {
        huggingface_login_command(None, false)?;
    }
    Ok(())
}

fn store_huggingface_token_in_local_store(token: &str) -> Result<()> {
    let credential_dir = huggingface_credential_dir();
    fs::create_dir_all(&credential_dir)
        .with_context(|| format!("failed to create {}", credential_dir.display()))?;
    set_private_permissions(&credential_dir)?;

    let key = local_encryption_key()?;
    let cipher = Aes256Gcm::new_from_slice(&key).expect("32-byte AES key");
    let mut nonce_bytes = [0_u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce_bytes), token.as_bytes())
        .map_err(|_| anyhow!("failed to encrypt Hugging Face token"))?;
    let mut payload = Vec::with_capacity(8 + nonce_bytes.len() + ciphertext.len());
    payload.extend_from_slice(b"AXHFENC1");
    payload.extend_from_slice(&nonce_bytes);
    payload.extend_from_slice(&ciphertext);
    let path = huggingface_encrypted_token_path();
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    set_private_permissions(&path)?;
    Ok(())
}

fn load_huggingface_token_from_local_store() -> Result<Option<String>> {
    let path = huggingface_encrypted_token_path();
    if !path.is_file() {
        return Ok(None);
    }
    let payload = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    if payload.len() < 20 || &payload[..8] != b"AXHFENC1" {
        bail!(
            "stored Hugging Face token at {} has an unknown format; run `axonal hf logout` then `axonal hf login`",
            path.display()
        );
    }
    let key = local_encryption_key()?;
    let cipher = Aes256Gcm::new_from_slice(&key).expect("32-byte AES key");
    let nonce = Nonce::from_slice(&payload[8..20]);
    let plaintext = cipher
        .decrypt(nonce, &payload[20..])
        .map_err(|_| anyhow!(
            "failed to decrypt the stored Hugging Face token at {}; the machine binding may have changed. Run `axonal hf logout` then `axonal hf login`",
            path.display()
        ))?;
    let token = String::from_utf8(plaintext).context("stored Hugging Face token is not valid UTF-8")?;
    Ok(Some(token.trim().to_string()))
}

fn local_encryption_key() -> Result<[u8; 32]> {
    let mut material = Vec::new();
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(value) = fs::read_to_string(path) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                material.extend_from_slice(trimmed.as_bytes());
                break;
            }
        }
    }
    if let Ok(user) = env::var("USER") {
        material.extend_from_slice(user.as_bytes());
    }
    if let Some(home) = dirs::home_dir() {
        material.extend_from_slice(home.as_os_str().to_string_lossy().as_bytes());
    }
    material.extend_from_slice(b"axonal-hf-local-store-v1");
    if material.is_empty() {
        bail!("could not derive local encryption key material");
    }
    let digest = Sha256::digest(material);
    let mut key = [0_u8; 32];
    key.copy_from_slice(&digest);
    Ok(key)
}

fn store_huggingface_token_in_tpm(token: &str) -> Result<()> {
    ensure_tpm_tools_available()?;
    let credential_dir = huggingface_credential_dir();
    fs::create_dir_all(&credential_dir)
        .with_context(|| format!("failed to create {}", credential_dir.display()))?;
    set_private_permissions(&credential_dir)?;

    let temp = TempWorkspace::new("axonal-hf-token")?;
    let token_path = temp.path().join("token.txt");
    let primary_ctx = temp.path().join("primary.ctx");
    let public_blob = temp.path().join("token.pub");
    let private_blob = temp.path().join("token.priv");

    fs::write(&token_path, token.as_bytes())
        .with_context(|| format!("failed to write {}", token_path.display()))?;
    set_private_permissions(&token_path)?;

    run_tpm_command(
        ProcessCommand::new("tpm2_createprimary")
            .arg("-C")
            .arg("o")
            .arg("-g")
            .arg("sha256")
            .arg("-G")
            .arg("rsa")
            .arg("-c")
            .arg(&primary_ctx),
        "tpm2_createprimary",
    )?;
    run_tpm_command(
        ProcessCommand::new("tpm2_create")
            .arg("-C")
            .arg(&primary_ctx)
            .arg("-G")
            .arg("keyedhash")
            .arg("-g")
            .arg("sha256")
            .arg("-u")
            .arg(&public_blob)
            .arg("-r")
            .arg(&private_blob)
            .arg("-i")
            .arg(&token_path),
        "tpm2_create",
    )?;

    fs::copy(&public_blob, huggingface_public_blob_path()).with_context(|| {
        format!(
            "failed to write {}",
            huggingface_public_blob_path().display()
        )
    })?;
    fs::copy(&private_blob, huggingface_private_blob_path()).with_context(|| {
        format!(
            "failed to write {}",
            huggingface_private_blob_path().display()
        )
    })?;
    set_private_permissions(&huggingface_public_blob_path())?;
    set_private_permissions(&huggingface_private_blob_path())?;
    flush_tpm_context(&primary_ctx);
    Ok(())
}

fn load_huggingface_token_from_tpm_store() -> Result<Option<String>> {
    let public_blob = huggingface_public_blob_path();
    let private_blob = huggingface_private_blob_path();
    if !public_blob.is_file() || !private_blob.is_file() {
        return Ok(None);
    }
    ensure_tpm_tools_available()?;
    let temp = TempWorkspace::new("axonal-hf-token-read")?;
    let primary_ctx = temp.path().join("primary.ctx");
    let sealed_ctx = temp.path().join("token.ctx");

    run_tpm_command(
        ProcessCommand::new("tpm2_createprimary")
            .arg("-C")
            .arg("o")
            .arg("-g")
            .arg("sha256")
            .arg("-G")
            .arg("rsa")
            .arg("-c")
            .arg(&primary_ctx),
        "tpm2_createprimary",
    )?;
    run_tpm_command(
        ProcessCommand::new("tpm2_load")
            .arg("-C")
            .arg(&primary_ctx)
            .arg("-u")
            .arg(&public_blob)
            .arg("-r")
            .arg(&private_blob)
            .arg("-c")
            .arg(&sealed_ctx),
        "tpm2_load",
    )?;
    let output = run_tpm_command_output(
        ProcessCommand::new("tpm2_unseal").arg("-c").arg(&sealed_ctx),
        "tpm2_unseal",
    )?;
    flush_tpm_context(&sealed_ctx);
    flush_tpm_context(&primary_ctx);
    let token = String::from_utf8(output.stdout).context("stored Hugging Face token is not valid UTF-8")?;
    Ok(Some(token.trim().to_string()))
}

fn ensure_tpm_tools_available() -> Result<()> {
    for tool in ["tpm2_createprimary", "tpm2_create", "tpm2_load", "tpm2_unseal", "tpm2_flushcontext"] {
        let status = ProcessCommand::new("which")
            .arg(tool)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| format!("failed to check for {tool}"))?;
        if !status.success() {
            bail!("{tool} is required for TPM-backed Hugging Face login");
        }
    }
    Ok(())
}

fn run_tpm_command(command: &mut ProcessCommand, name: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to launch {name}"))?;
    if output.status.success() {
        return Ok(());
    }
    bail!(format_tpm_command_error(name, &output.stderr))
}

fn run_tpm_command_output(command: &mut ProcessCommand, name: &str) -> Result<std::process::Output> {
    let output = command
        .output()
        .with_context(|| format!("failed to launch {name}"))?;
    if output.status.success() {
        return Ok(output);
    }
    bail!(format_tpm_command_error(name, &output.stderr))
}

fn format_tpm_command_error(name: &str, stderr: &[u8]) -> String {
    let details = String::from_utf8_lossy(stderr).trim().to_string();
    if details.is_empty() {
        format!(
            "{name} failed. Ensure this user can access /dev/tpmrm0 or /dev/tpm0, usually via the `tss` group."
        )
    } else {
        format!(
            "{name} failed: {details}. Ensure this user can access /dev/tpmrm0 or /dev/tpm0, usually via the `tss` group."
        )
    }
}

fn flush_tpm_context(path: &Path) {
    let _ = ProcessCommand::new("tpm2_flushcontext")
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn set_private_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let mode = if path.is_dir() { 0o700 } else { 0o600 };
        fs::set_permissions(path, fs::Permissions::from_mode(mode))
            .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    }
    Ok(())
}

fn convert_command(
    source: &str,
    local: bool,
    revision: &str,
    source_format: Option<&str>,
    name: Option<&str>,
    library: Option<&Path>,
    replace: bool,
    quantization: &str,
    group_size: Option<usize>,
    outlier_sigma: f64,
    no_vq: bool,
) -> Result<PathBuf> {
    let model_name = normalize_model_name(name.unwrap_or(&default_convert_name(source, local)));
    let library_root = library
        .map(PathBuf::from)
        .unwrap_or_else(default_models_dir);
    let install_path = library_root.join(&model_name);
    let candidate = PathBuf::from(source);
    let mut workspace = None;
    eprintln!(
        "[axonal] convert: source={} target={}",
        source,
        install_path.display()
    );
    let source_path = if local || candidate.exists() {
        eprintln!("[axonal] convert: using local input {}", candidate.display());
        candidate
    } else {
        let temp = TempWorkspace::new("axonal-convert")?;
        let checkout = temp.path().join("source");
        maybe_prompt_for_huggingface_login()?;
        eprintln!(
            "[axonal] convert: downloading Hugging Face repo {} @ {}",
            source, revision
        );
        download_huggingface_repo(source, revision, &checkout)?;
        workspace = Some(temp);
        checkout
    };
    let canonical_source = if source_path.exists() {
        fs::canonicalize(&source_path).unwrap_or(source_path)
    } else {
        source_path
    };
    let (pack_input, resolved_source_format) =
        resolve_convert_input(&canonical_source, source_format)?;
    eprintln!(
        "[axonal] convert: packing {} as {}",
        pack_input.display(),
        resolved_source_format
    );
    run_axon_pack_pack(
        &pack_input,
        &resolved_source_format,
        &model_name,
        &library_root,
        replace,
        quantization,
        group_size,
        outlier_sigma,
        no_vq,
    )?;
    drop(workspace);
    Ok(install_path)
}

fn default_host() -> String {
    env::var("AXONAL_HOST")
        .or_else(|_| env::var("OLLAMA_HOST"))
        .unwrap_or_else(|_| "http://127.0.0.1:11435".to_string())
}

fn default_models_dir() -> PathBuf {
    if let Ok(path) = env::var("AXONAL_MODELS") {
        return PathBuf::from(path);
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".axonal").join("models");
    }
    PathBuf::from("./models")
}

fn default_convert_name(source: &str, local: bool) -> String {
    let candidate = Path::new(source);
    if local || candidate.exists() {
        if candidate.is_dir() {
            return candidate
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string());
        }
        if let Some(stem) = candidate.file_stem() {
            return stem.to_string_lossy().to_string();
        }
    }
    source
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("model")
        .to_string()
}

fn normalize_model_name(name: &str) -> String {
    let mut output = String::new();
    let mut last_was_dash = false;
    for ch in name.trim().chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() || lowered == '.' || lowered == '_' || lowered == '-' {
            output.push(lowered);
            last_was_dash = false;
        } else if !last_was_dash {
            output.push('-');
            last_was_dash = true;
        }
    }
    let trimmed = output.trim_matches(|ch| ch == '-' || ch == '.' || ch == '_');
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed.to_string()
    }
}

fn resolve_convert_input(path: &Path, source_format: Option<&str>) -> Result<(PathBuf, String)> {
    match source_format {
        Some("hf") => {
            if !is_hf_model_dir(path) {
                bail!("{} is not a Hugging Face safetensors model directory", path.display());
            }
            Ok((path.to_path_buf(), "hf".to_string()))
        }
        Some("gguf") => {
            let gguf = resolve_gguf_input(path)?;
            Ok((gguf, "gguf".to_string()))
        }
        Some(other) => bail!("unsupported source format {other}"),
        None => infer_convert_input(path),
    }
}

fn infer_convert_input(path: &Path) -> Result<(PathBuf, String)> {
    if path.is_file() {
        if is_gguf_file(path) {
            return Ok((path.to_path_buf(), "gguf".to_string()));
        }
        bail!("local file inputs must be .gguf files; use a directory for safetensors models");
    }
    if is_hf_model_dir(path) {
        return Ok((path.to_path_buf(), "hf".to_string()));
    }
    let gguf = resolve_gguf_input(path)?;
    Ok((gguf, "gguf".to_string()))
}

fn is_hf_model_dir(path: &Path) -> bool {
    if !path.is_dir() || !path.join("config.json").is_file() {
        return false;
    }
    fs::read_dir(path)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .any(|entry| {
            entry
                .file_name()
                .and_then(|value| value.to_str())
                .map(|name| name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json"))
                .unwrap_or(false)
        })
}

fn resolve_gguf_input(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        if is_gguf_file(path) {
            return Ok(path.to_path_buf());
        }
        bail!("{} is not a .gguf file", path.display());
    }
    let ggufs = fs::read_dir(path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|entry| is_gguf_file(entry))
        .collect::<Vec<_>>();
    match ggufs.len() {
        1 => Ok(ggufs.into_iter().next().unwrap()),
        0 => bail!("could not find a safetensors model directory or gguf file in {}", path.display()),
        _ => bail!("multiple gguf files found in {}; pass a direct .gguf path", path.display()),
    }
}

fn is_gguf_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|value| value.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
}

fn http_client() -> Client {
    Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("failed to build HTTP client")
}

fn huggingface_headers() -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    if let Some(token) = huggingface_auth_token()? {
        let value = HeaderValue::from_str(&format!("Bearer {token}")).context("invalid HF token")?;
        headers.insert(AUTHORIZATION, value);
    }
    Ok(headers)
}

fn download_huggingface_repo(repo_id: &str, revision: &str, target_dir: &Path) -> Result<()> {
    fs::create_dir_all(target_dir)
        .with_context(|| format!("failed to create {}", target_dir.display()))?;
    let client = http_client();
    let headers = huggingface_headers()?;
    let info_url = huggingface_model_info_url(repo_id, revision)?;
    let response = client
        .get(info_url)
        .headers(headers.clone())
        .send()
        .context("failed to query Hugging Face model metadata")?;
    let response = require_hf_success(response, repo_id)?;
    let info: HuggingFaceModelInfo = response
        .json()
        .context("failed to parse Hugging Face model metadata")?;
    let files = info
        .siblings
        .into_iter()
        .map(|entry| entry.rfilename)
        .filter(|name| should_download_hf_file(name))
        .collect::<Vec<_>>();
    if files.is_empty() {
        bail!("no supported safetensors, gguf, config, or tokenizer files found in {repo_id}");
    }
    eprintln!(
        "[axonal] convert: downloading {} file(s) from {}",
        files.len(),
        repo_id
    );
    let total_files = files.len();
    for (index, file_name) in files.into_iter().enumerate() {
        download_huggingface_file(
            &client,
            &headers,
            repo_id,
            revision,
            &file_name,
            target_dir,
            index + 1,
            total_files,
        )?;
    }
    eprintln!("[axonal] convert: download complete");
    Ok(())
}

fn huggingface_model_info_url(repo_id: &str, revision: &str) -> Result<Url> {
    let mut url = Url::parse("https://huggingface.co/api/models")
        .context("failed to construct Hugging Face API URL")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("failed to update Hugging Face API URL"))?;
        for segment in repo_id.split('/') {
            segments.push(segment);
        }
        segments.push("revision");
        segments.push(revision);
    }
    Ok(url)
}

fn huggingface_resolve_url(repo_id: &str, revision: &str, file_name: &str) -> Result<Url> {
    let mut url =
        Url::parse("https://huggingface.co").context("failed to construct Hugging Face resolve URL")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("failed to update Hugging Face resolve URL"))?;
        for segment in repo_id.split('/') {
            segments.push(segment);
        }
        segments.push("resolve");
        segments.push(revision);
        for segment in file_name.split('/') {
            segments.push(segment);
        }
    }
    Ok(url)
}

fn require_hf_success(response: Response, repo_id: &str) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    if status.as_u16() == 401 || status.as_u16() == 403 {
        bail!("failed to access Hugging Face repo {repo_id}; set HF_TOKEN for private or gated models");
    }
    bail!("failed to access Hugging Face repo {repo_id}: HTTP {status}")
}

fn download_huggingface_file(
    client: &Client,
    headers: &HeaderMap,
    repo_id: &str,
    revision: &str,
    file_name: &str,
    target_dir: &Path,
    file_index: usize,
    total_files: usize,
) -> Result<()> {
    let url = huggingface_resolve_url(repo_id, revision, file_name)?;
    let response = client
        .get(url)
        .headers(headers.clone())
        .send()
        .with_context(|| format!("failed to download {file_name} from {repo_id}"))?;
    let response = require_hf_success(response, repo_id)?;
    let total_bytes = response.content_length();
    match total_bytes {
        Some(total_bytes) => eprintln!(
            "[axonal] download {}/{}: {} ({})",
            file_index,
            total_files,
            file_name,
            format_byte_size(total_bytes)
        ),
        None => eprintln!(
            "[axonal] download {}/{}: {}",
            file_index,
            total_files,
            file_name
        ),
    }
    let destination = target_dir.join(file_name);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut file = File::create(&destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    copy_response_with_progress(response, &mut file, file_name, total_bytes)
        .with_context(|| format!("failed to write {}", destination.display()))?;
    Ok(())
}

fn copy_response_with_progress(
    mut response: Response,
    file: &mut File,
    file_name: &str,
    total_bytes: Option<u64>,
) -> Result<()> {
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut written = 0_u64;
    let started = Instant::now();
    let mut last_log = Instant::now();
    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        written += bytes_read as u64;
        if last_log.elapsed() >= Duration::from_secs(2) {
            match total_bytes {
                Some(total_bytes) if total_bytes > 0 => eprintln!(
                    "[axonal]   progress: {} / {} ({:.1}%)",
                    format_byte_size(written),
                    format_byte_size(total_bytes),
                    (written as f64 * 100.0) / total_bytes as f64
                ),
                _ => eprintln!("[axonal]   progress: {}", format_byte_size(written)),
            }
            last_log = Instant::now();
        }
    }
    match total_bytes {
        Some(total_bytes) if total_bytes > 0 => eprintln!(
            "[axonal]   complete: {} in {:.1}s ({})",
            file_name,
            started.elapsed().as_secs_f64(),
            format_byte_size(total_bytes)
        ),
        _ => eprintln!(
            "[axonal]   complete: {} in {:.1}s ({})",
            file_name,
            started.elapsed().as_secs_f64(),
            format_byte_size(written)
        ),
    }
    Ok(())
}

fn format_byte_size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn should_download_hf_file(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    lower.ends_with(".safetensors")
        || lower.ends_with(".safetensors.index.json")
        || lower.ends_with(".gguf")
        || lower.ends_with("config.json")
        || lower.ends_with("generation_config.json")
        || lower.ends_with("tokenizer.json")
        || lower.ends_with("tokenizer_config.json")
        || lower.ends_with("special_tokens_map.json")
        || lower.ends_with("added_tokens.json")
        || lower.ends_with("processor_config.json")
        || lower.ends_with("preprocessor_config.json")
        || lower.ends_with("feature_extractor_config.json")
        || lower.ends_with("image_processor_config.json")
        || lower.ends_with("video_preprocessor_config.json")
        || lower.ends_with("chat_template.jinja")
        || lower.ends_with("merges.txt")
        || lower.ends_with("vocab.txt")
        || lower.ends_with(".model")
        || lower.ends_with(".tiktoken")
        || lower.ends_with("/vocab.json")
        || lower == "vocab.json"
}

fn axon_pack_repo() -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = env::var("AXON_PACK_REPO") {
        candidates.push(PathBuf::from(path));
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent() {
        candidates.push(parent.join("axon-pack"));
    }
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("axon-pack"));
        if let Some(parent) = cwd.parent() {
            candidates.push(parent.join("axon-pack"));
        }
    }
    for candidate in candidates {
        if candidate.join("python").join("axon_pack").join("cli.py").is_file() {
            return Ok(candidate);
        }
    }
    bail!("could not locate axon-pack; set AXON_PACK_REPO to the axon-pack checkout")
}

fn run_axon_pack_pack(
    input: &Path,
    source_format: &str,
    name: &str,
    library_root: &Path,
    replace: bool,
    quantization: &str,
    group_size: Option<usize>,
    outlier_sigma: f64,
    no_vq: bool,
) -> Result<()> {
    let repo = axon_pack_repo()?;
    let python = env::var("AXONAL_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut command = ProcessCommand::new(python);
    let python_root = repo.join("python");
    let pythonpath = match env::var_os("PYTHONPATH") {
        Some(existing) => env::join_paths([python_root.as_os_str(), existing.as_os_str()])
            .context("failed to compose PYTHONPATH for axon-pack")?,
        None => python_root.into_os_string(),
    };
    command
        .current_dir(&repo)
        .env("PYTHONPATH", pythonpath)
        .arg("-m")
        .arg("axon_pack.cli")
        .arg("pack")
        .arg("--input")
        .arg(input)
        .arg("--source-format")
        .arg(source_format)
        .arg("--name")
        .arg(name)
        .arg("--library")
        .arg(library_root)
        .arg("--quantization")
        .arg(quantization)
        .arg("--outlier-sigma")
        .arg(outlier_sigma.to_string());
    if replace {
        command.arg("--replace");
    }
    if let Some(group_size) = group_size {
        command.arg("--group-size").arg(group_size.to_string());
    }
    if no_vq {
        command.arg("--no-vq");
    }
    eprintln!(
        "[axonal] convert: invoking axon-pack for {}",
        input.display()
    );
    let status = command.status().context("failed to launch axon-pack")?;
    if !status.success() {
        bail!("axon-pack failed with status {status}");
    }
    Ok(())
}

fn normalize_host(host: Option<&str>) -> String {
    let host = host.map(ToOwned::to_owned).unwrap_or_else(default_host);
    if host.starts_with("http://") || host.starts_with("https://") {
        host
    } else {
        format!("http://{host}")
    }
}

fn run_command(
    model: &str,
    prompt: &str,
    host: Option<&str>,
    backend: Option<&str>,
    options: RunOptions,
) -> Result<GenerationPreview> {
    if let Ok(response) = run_remote(model, prompt, host, backend, &options) {
        return Ok(response);
    }
    if host.is_none() {
        let _ = start_local_server_for_model(model);
        if let Ok(response) = run_remote(model, prompt, host, backend, &options) {
            return Ok(response);
        }
    }
    let bundle = resolve_local_bundle(model)?;
    run_model_with_options(&bundle, prompt, backend, options)
}

fn show_command(model: &str, host: Option<&str>, backend: Option<&str>) -> Result<ShowResponse> {
    if let Ok(response) = show_remote(model, host, backend) {
        return Ok(response);
    }
    let bundle = resolve_local_bundle(model)?;
    show_bundle(&bundle, backend)
}

fn list_command(host: Option<&str>) -> Result<TagsResponse> {
    if let Some(host) = host {
        return list_remote(Some(host));
    }
    let registry = scan_registry(&default_models_dir())?;
    Ok(TagsResponse {
        models: tags_from_registry(&registry),
    })
}

fn ps_command(host: Option<&str>) -> Result<PsResponse> {
    match ps_remote(host) {
        Ok(response) => Ok(response),
        Err(_) if host.is_none() => Ok(PsResponse { models: Vec::new() }),
        Err(error) => Err(error),
    }
}

fn run_remote(
    model: &str,
    prompt: &str,
    host: Option<&str>,
    backend: Option<&str>,
    options: &RunOptions,
) -> Result<GenerationPreview> {
    let model_path = bundle_request_path(model);
    let response = http_client()
        .post(format!("{}/api/generate", normalize_host(host)))
        .json(&json!({
            "model": model,
            "path": model_path,
            "prompt": prompt,
            "backend": backend,
            "num_predict": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "seed": options.seed,
            "raw": options.raw_prompt
        }))
        .send()?;
    if !response.status().is_success() {
        bail!("server returned {}", response.status());
    }
    Ok(response.json()?)
}

fn show_remote(model: &str, host: Option<&str>, backend: Option<&str>) -> Result<ShowResponse> {
    let model_path = bundle_request_path(model);
    let response = http_client()
        .post(format!("{}/api/show", normalize_host(host)))
        .json(&json!({ "name": model, "path": model_path, "backend": backend }))
        .send()?;
    if !response.status().is_success() {
        bail!("server returned {}", response.status());
    }
    Ok(response.json()?)
}

fn list_remote(host: Option<&str>) -> Result<TagsResponse> {
    let response = http_client().get(format!("{}/api/tags", normalize_host(host))).send()?;
    if !response.status().is_success() {
        bail!("server returned {}", response.status());
    }
    Ok(response.json()?)
}

fn ps_remote(host: Option<&str>) -> Result<PsResponse> {
    let response = http_client().get(format!("{}/api/ps", normalize_host(host))).send()?;
    if !response.status().is_success() {
        bail!("server returned {}", response.status());
    }
    Ok(response.json()?)
}

fn resolve_local_bundle(model: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(model);
    if candidate.is_dir() && candidate.join("manifest.json").is_file() {
        return Ok(candidate);
    }
    let registry = scan_registry(&default_models_dir())?;
    resolve_registered_bundle(&registry, model)
}

fn resolve_registered_bundle(models: &[RegisteredModel], model: &str) -> Result<PathBuf> {
    let registered = get_registered_model(models, model)
        .ok_or_else(|| anyhow!("unknown model {model}"))?;
    Ok(registered.path.clone())
}

fn resolve_generate_bundle(models: &[RegisteredModel], path: Option<&str>, model: &str) -> Result<PathBuf> {
    if let Some(path) = path {
        let bundle = PathBuf::from(path);
        if bundle.is_dir() && bundle.join("manifest.json").is_file() {
            return Ok(bundle);
        }
        bail!("unknown bundle path {path}");
    }
    resolve_registered_bundle(models, model)
}

fn lookup_show(
    models: &[RegisteredModel],
    name: Option<&str>,
    path: Option<&str>,
    backend: Option<&str>,
) -> Result<ShowResponse> {
    if let Some(path) = path {
        return show_bundle(Path::new(path), backend);
    }
    if let Some(name) = name {
        let bundle = resolve_registered_bundle(models, name)?;
        return show_bundle(&bundle, backend);
    }
    Err(anyhow!("request must include name or path"))
}

fn bundle_request_path(model: &str) -> Option<String> {
    let candidate = PathBuf::from(model);
    if candidate.is_dir() && candidate.join("manifest.json").is_file() {
        Some(candidate.display().to_string())
    } else {
        None
    }
}

fn start_local_server_for_model(_model: &str) -> Result<()> {
    let host = normalize_host(None);
    if list_remote(Some(&host)).is_ok() {
        return Ok(());
    }

    let models_root = default_models_dir();
    let url = Url::parse(&host).context("failed to parse local server URL")?;
    let host_name = url
        .host_str()
        .ok_or_else(|| anyhow!("local server URL is missing a host"))?;
    let port = url.port_or_known_default().unwrap_or(11435);
    let exe = env::current_exe().context("failed to locate axonal executable")?;

    ProcessCommand::new(exe)
        .arg("serve")
        .arg("--models")
        .arg(models_root)
        .arg("--host")
        .arg(host_name)
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to start local axonal server")?;

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(10) {
        if list_remote(Some(&host)).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!("local axonal server did not become ready in time")
}

fn render_run_output(response: &GenerationPreview, verbose: bool, _nowordwrap: bool) -> Result<()> {
    if verbose {
        println!("{}", serde_json::to_string_pretty(response)?);
        return Ok(());
    }
    if !response.response.is_empty() {
        println!("{}", response.response);
    } else {
        println!("{}", response.message);
        println!(
            "backend: {} {} ({})",
            response.backend.backend, response.backend.hw_hint, response.backend.kernel
        );
    }
    Ok(())
}

fn format_parameter_summary(active: Option<u64>, total: Option<u64>) -> String {
    match (active, total) {
        (Some(active), Some(total)) if active != total => {
            format!("{} active / {} total", format_params(active), format_params(total))
        }
        (_, Some(total)) => format_params(total),
        (Some(active), None) => format!("{} active", format_params(active)),
        (None, None) => "unknown".to_string(),
    }
}

fn format_params(value: u64) -> String {
    const BILLION: f64 = 1_000_000_000.0;
    const MILLION: f64 = 1_000_000.0;
    if value >= 1_000_000_000 {
        format!("{:.1}B", value as f64 / BILLION)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / MILLION)
    } else {
        value.to_string()
    }
}

fn print_tags(models: &[axonal::ModelTag]) {
    println!("{:<24} {:<16} {:<20} {}", "NAME", "FAMILY", "ARCH", "SIZE");
    for model in models {
        println!(
            "{:<24} {:<16} {:<20} {}",
            model.name, model.model_family, model.architecture, model.file_size
        );
    }
}

fn print_processes(models: &[axonal::RuntimeProcessInfo]) {
    if models.is_empty() {
        println!("No loaded models.");
        return;
    }
    println!(
        "{:<24} {:<8} {:<8} {:<20} {:<6} {}",
        "NAME", "STATE", "BACKEND", "ARCH", "REQS", "LAST_USED"
    );
    for model in models {
        println!(
            "{:<24} {:<8} {:<8} {:<20} {:<6} {}",
            model.name,
            model.state,
            model.backend,
            model.architecture,
            model.request_count,
            model.last_used_unix
        );
    }
}
