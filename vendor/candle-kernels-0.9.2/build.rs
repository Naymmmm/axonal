use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

const CUDA_KERNELS: &[&str] = &[
    "src/affine.cu",
    "src/binary.cu",
    "src/cast.cu",
    "src/conv.cu",
    "src/fill.cu",
    "src/indexing.cu",
    "src/quantized.cu",
    "src/reduce.cu",
    "src/sort.cu",
    "src/ternary.cu",
    "src/unary.cu",
];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);
    build_cubins(&out_dir);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}

fn build_cubins(out_dir: &PathBuf) {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

    let compute_cap = compute_cap();
    let cuda_include_dir = cuda_include_dir().expect("failed to locate CUDA include directory");
    let nvcc = nvcc_path();
    let ccbin_env = std::env::var("NVCC_CCBIN").ok();
    let cubin_rs_path = out_dir.join("cubin.rs");
    let mut cubin_rs = File::create(&cubin_rs_path).expect("failed to create cubin.rs");

    for kernel in CUDA_KERNELS {
        println!("cargo:rerun-if-changed={kernel}");
        let kernel_path = PathBuf::from(kernel);
        let stem = kernel_path
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("kernel filename must be valid utf-8");
        let output = out_dir.join(format!("{stem}.cubin"));
        if should_rebuild(&kernel_path, &output) {
            let mut command = Command::new(&nvcc);
            command
                .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                .arg("--cubin")
                .args(["--default-stream", "per-thread"])
                .arg("-o")
                .arg(&output)
                .arg("--expt-relaxed-constexpr")
                .arg("-std=c++17")
                .arg("-O3")
                .arg(format!("-I{}", cuda_include_dir.display()))
                .arg(&kernel_path);
            if let Some(ccbin_path) = &ccbin_env {
                command
                    .arg("-allow-unsupported-compiler")
                    .args(["-ccbin", ccbin_path]);
            }
            let result = command
                .output()
                .expect("nvcc failed to start while building candle cubins");
            assert!(
                result.status.success(),
                "nvcc error while compiling {kernel} to cubin:\n\n# stdout\n{}\n\n# stderr\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }

        writeln!(
            cubin_rs,
            r#"pub const {}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{}.cubin"));"#,
            stem.to_uppercase(),
            stem
        )
        .expect("failed to write cubin constant");
    }
}

fn should_rebuild(input: &PathBuf, output: &PathBuf) -> bool {
    let Ok(output_meta) = std::fs::metadata(output) else {
        return true;
    };
    let Ok(output_modified) = output_meta.modified() else {
        return true;
    };
    let Ok(input_modified) = std::fs::metadata(input).and_then(|meta| meta.modified()) else {
        return true;
    };
    input_modified.duration_since(output_modified).is_ok()
}

fn compute_cap() -> usize {
    if let Ok(compute_cap) = std::env::var("CUDA_COMPUTE_CAP") {
        return compute_cap
            .parse::<usize>()
            .expect("failed to parse CUDA_COMPUTE_CAP");
    }

    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv")
        .output()
        .expect("nvidia-smi failed while detecting compute capability");
    let stdout = String::from_utf8(output.stdout).expect("nvidia-smi output was not valid utf-8");
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("compute_cap"));
    lines
        .next()
        .expect("missing compute capability row")
        .replace('.', "")
        .parse::<usize>()
        .expect("failed to parse compute capability")
}

fn cuda_include_dir() -> Option<PathBuf> {
    for env_var in ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR", "CUDA_HOME"] {
        if let Ok(path) = std::env::var(env_var) {
            let include = PathBuf::from(path).join("include");
            if include.join("cuda.h").is_file() {
                return Some(include);
            }
        }
    }

    for root in ["/usr/local/cuda", "/opt/cuda", "/usr"] {
        let include = PathBuf::from(root).join("include");
        if include.join("cuda.h").is_file() {
            return Some(include);
        }
    }
    None
}

fn nvcc_path() -> PathBuf {
    for env_var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(path) = std::env::var(env_var) {
            let nvcc = PathBuf::from(path).join("bin").join("nvcc");
            if nvcc.is_file() {
                return nvcc;
            }
        }
    }
    PathBuf::from("nvcc")
}
