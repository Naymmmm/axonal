# axonal

`axonal` is the local runtime, model library, and Ollama-style CLI/API layer for AXON bundles.

It can:

- inspect and validate AXON bundles
- convert Hugging Face repos or local `safetensors` / `gguf` sources into AXON bundles via `axon-pack`
- install converted bundles into a shared local model library
- list installed models with `axonal ls`
- show loaded runtimes with `axonal ps`
- run local generation from AXON bundles on CPU or CUDA
- serve an Ollama-style local HTTP API
- resolve LoRA delta bundles against a local AXON base bundle at runtime

Current generation support is focused on causal LMs, with working runtime paths for Llama-family and Qwen2-family bundles.

## Related Repos

- AXON spec: `github.com/Naymmmm/axon`
- AXON packer: this repo uses `axon-pack` for source import and bundle creation

## Installation

### Rust

`axonal` is installable as a normal Cargo binary:

```bash
cargo install --path . --bin axonal
```

By default the crate enables CUDA support. For a CPU-only build:

```bash
cargo install --path . --bin axonal --no-default-features
```

### From Source

```bash
cargo build --release
./target/release/axonal --help
```

## Quickstart

### 1. Log into Hugging Face if needed

```bash
axonal hf login
```

Default token storage uses a local encrypted file under `~/.axonal/credentials/huggingface/token.enc`.

Use TPM-backed storage only if you explicitly want it:

```bash
axonal hf login --tpm
```

### 2. Convert a model into the local AXON library

Remote Hugging Face repo:

```bash
axonal convert Qwen/Qwen2.5-0.5B-Instruct --name qwen2_5_0_5b
```

Local source directory or GGUF file:

```bash
axonal convert /path/to/model --local --name local-model --pack-jobs 8
```

### 3. Inspect and run it

```bash
axonal ls
axonal show qwen2_5_0_5b
axonal run qwen2_5_0_5b "Hello"
```

### 4. Serve the local API

```bash
axonal serve
```

The default API host is `127.0.0.1:11435`.

## Commands

- `axonal inspect`: validate a bundle path directly
- `axonal convert`: download or pack a source model into the local AXON library
- `axonal run`: generate text from a named bundle or bundle path
- `axonal show`: inspect model metadata
- `axonal list` / `axonal ls`: list installed bundles
- `axonal ps`: list cached running models in the local daemon
- `axonal serve`: start the local API server
- `axonal hf login|logout|status`: manage Hugging Face credentials

More detail: [docs/cli.md](/home/oscar/Repos/axonal/docs/cli.md)

## Model Library

By default `axonal` uses:

- `$AXONAL_MODELS` when set
- otherwise `~/.axonal/models`

`axonal convert` installs bundles there by default, and `axonal ls` scans the same location.

Useful performance flags for conversion:

- `--download-jobs`: parallel Hugging Face file downloads
- `--pack-jobs`: parallel tensor packing workers inside `axon-pack`
- `--pack-gpu`: enable CUDA-backed quantization in `axon-pack` when PyTorch/CUDA is available

LoRA delta bundles are resolved against a base AXON bundle by searching:

- the delta bundle directory
- sibling bundle directories
- the shared model library

## HTTP API

Current API surface:

- `GET /api/tags`
- `GET /api/ps`
- `POST /api/show`
- `POST /api/generate`

The API is intentionally close to an Ollama-style local workflow, but AXON-native.

## Runtime Notes

- CPU execution includes a native AXON decoder path.
- CUDA execution uses a cached Candle runtime and preloads merged tensors for supported architectures.
- LoRA delta bundles are merged at runtime:
  - CPU path applies low-rank updates during matvecs.
  - CUDA path pre-merges the low-rank update at model load time.

## Current Limits

- `tokenizer.json` is still required for actual inference.
- Custom Hugging Face Python code is not executed.
- Runtime architecture support is narrower than the full AXON spec today.
- CUDA kernel work exists, but not every AXON quantization path is kernel-native yet.

## Development

```bash
cargo test
cargo build --release
```

`cargo fmt` requires `rustfmt` to be installed in the current toolchain.
