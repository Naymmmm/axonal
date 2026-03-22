# axonal CLI

## Overview

`axonal` is the local command surface for:

- converting source models into AXON bundles
- managing the local model library
- running local inference
- exposing a local HTTP API

## Convert

Remote Hugging Face repo:

```bash
axonal convert Qwen/Qwen2.5-0.5B-Instruct --name qwen2_5_0_5b
```

Local source:

```bash
axonal convert /path/to/model --local --name my-model
```

Useful flags:

- `--local`: treat the source as a local folder or GGUF file
- `--revision`: Hugging Face revision for remote downloads
- `--source-format hf|gguf`: force source format
- `--name`: install name inside the model library
- `--library`: override the library root
- `--replace`: overwrite an existing installed bundle
- `--quantization none|auto|mxq`
- `--group-size`
- `--outlier-sigma`
- `--no-vq`
- `--download-jobs`: parallelize remote Hugging Face downloads
- `--pack-jobs`: parallelize tensor packing work in `axon-pack`
- `--pack-gpu`: enable CUDA-backed quantization when the packer has PyTorch/CUDA available

## Run

```bash
axonal run MODEL "Prompt text"
```

Useful flags:

- `--backend cpu|cuda`
- `--max-tokens`
- `--temperature`
- `--top-p`
- `--top-k`
- `--seed`
- `--raw`: skip chat templating
- `--host`: run against an existing daemon instead of local fallback
- `--verbose`

## Inspect And Registry Commands

```bash
axonal inspect --bundle /path/to/bundle
axonal show MODEL
axonal ls
axonal ps
```

- `inspect` works on a direct bundle path
- `show` works on a named installed model or bundle path
- `ls` lists installed bundles
- `ps` lists loaded daemon runtimes and cache state

## Serve

```bash
axonal serve --models ~/.axonal/models --host 127.0.0.1 --port 11435
```

API endpoints:

- `GET /api/tags`
- `GET /api/ps`
- `POST /api/show`
- `POST /api/generate`

## Hugging Face Auth

```bash
axonal hf login
axonal hf status
axonal hf logout
```

Default login stores the token in a local encrypted file.

Use `--tpm` only if you explicitly want TPM-backed token sealing:

```bash
axonal hf login --tpm
```

## Model Resolution

When you pass `MODEL`, `axonal` resolves it from:

- the shared model library
- a direct bundle path
- for LoRA delta bundles, the base AXON bundle is searched in:
  - the delta bundle directory
  - sibling directories
  - the shared model library
