use crate::BundleSummary;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSelection {
    pub backend: String,
    pub device: String,
    pub hw_hint: String,
    pub kernel: String,
    pub kernel_source: Option<String>,
}

pub fn select_backend(summary: &BundleSummary, preferred: Option<&str>) -> BackendSelection {
    match preferred.unwrap_or("auto") {
        "cpu" => cpu_backend(summary),
        "cuda" => cuda_backend(summary)
            .or_else(generic_cuda_backend)
            .unwrap_or_else(|| cpu_backend(summary)),
        _ => cuda_backend(summary)
            .or_else(generic_cuda_backend)
            .unwrap_or_else(|| cpu_backend(summary)),
    }
}

pub fn execution_backend(
    summary: &BundleSummary,
    preferred: Option<&str>,
) -> (BackendSelection, Option<String>) {
    let selected = select_backend(summary, preferred);
    if selected.backend == "cpu" {
        return (selected, None);
    }
    (selected, None)
}

pub fn cuda_kernel_source() -> &'static str {
    include_str!("../kernels/cuda/mxq_kernels.cu")
}

pub fn cpu_backend(summary: &BundleSummary) -> BackendSelection {
    if let Some(hint) = summary.metadata.hw_hints.get("cpu_avx512_vnni") {
        return BackendSelection {
            backend: "cpu".to_string(),
            device: "cpu".to_string(),
            hw_hint: "cpu_avx512_vnni".to_string(),
            kernel: hint.kernel.clone(),
            kernel_source: None,
        };
    }
    if let Some(hint) = summary.metadata.hw_hints.get("cpu_neon") {
        return BackendSelection {
            backend: "cpu".to_string(),
            device: "cpu".to_string(),
            hw_hint: "cpu_neon".to_string(),
            kernel: hint.kernel.clone(),
            kernel_source: None,
        };
    }
    BackendSelection {
        backend: "cpu".to_string(),
        device: "cpu".to_string(),
        hw_hint: "cpu_generic".to_string(),
        kernel: "axonal_cpu_generic".to_string(),
        kernel_source: None,
    }
}

fn cuda_backend(summary: &BundleSummary) -> Option<BackendSelection> {
    let device_name = detect_cuda_device()?;
    let hint_id = cuda_hint_for_device(summary, &device_name)?;
    let hint = summary.metadata.hw_hints.get(&hint_id)?;
    Some(BackendSelection {
        backend: "cuda".to_string(),
        device: device_name,
        hw_hint: hint_id,
        kernel: hint.kernel.clone(),
        kernel_source: Some("kernels/cuda/mxq_kernels.cu".to_string()),
    })
}

fn generic_cuda_backend() -> Option<BackendSelection> {
    let device_name = detect_cuda_device()?;
    Some(BackendSelection {
        backend: "cuda".to_string(),
        device: device_name,
        hw_hint: "cuda_generic".to_string(),
        kernel: "axonal_cuda_candle".to_string(),
        kernel_source: None,
    })
}

fn detect_cuda_device() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find(|line| !line.trim().is_empty()).map(|line| line.trim().to_string())
}

fn cuda_hint_for_device(summary: &BundleSummary, device_name: &str) -> Option<String> {
    let upper = device_name.to_ascii_uppercase();
    for (needle, hint_id) in [("H100", "cuda_sm90"), ("H200", "cuda_sm90"), ("4090", "cuda_sm89"), ("4080", "cuda_sm89"), ("A100", "cuda_sm80"), ("A10", "cuda_sm80")] {
        if upper.contains(needle) && summary.metadata.hw_hints.contains_key(hint_id) {
            return Some(hint_id.to_string());
        }
    }
    for hint_id in ["cuda_sm90", "cuda_sm89", "cuda_sm80"] {
        if summary.metadata.hw_hints.contains_key(hint_id) {
            return Some(hint_id.to_string());
        }
    }
    None
}
