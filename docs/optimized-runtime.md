# Canonical optimized runtime

WordVoice production workloads must use the native TensorRT decoder path when
the configured GPU profile is available. The eager PyTorch decoder is a
qualification and development path; it is not an automatic production
fallback. If the optimized engine cannot be loaded or its manifest does not
match the checkpoint, TensorRT, or GPU, fail closed and report the original
gate error.

## RTX 3060 profile

The current canonical profile is:

```text
revision: starline-rtx3060-native-trt-flow-bf16-prefix-reuse-v33-combined-logit-mask
decoder: TensorRT, FP32, token-major-flat-v1 KV cache
flow: PyTorch CUDA BF16, 6 diffusion steps
GPU: NVIDIA GeForce RTX 3060 (compute capability 8.6)
TensorRT: 11.0.0.114
```

The decoder engine is generated outside Git because it is a large,
GPU-specific binary. The runtime directory must contain `decoder.plan` and its
matching `manifest.json`; the adapter verifies both hashes before inference.
On Alexander's Windows workstation, the promoted durable runtime directory is
`D:\\ainotebook-models\\wordvoice-native-tensorrt\\rtx3060-sm86-trt11-flat-cache-v25`.
The exact qualified engine for this profile has plan SHA-256
`c076b57bd7c99579381e8f0901936fc22345cb2cf8384b83ff5c644ff9cdc81f` and
manifest SHA-256
`776f5865a65d276b154e6b08c4bfb86f32666872c3f16b97396908ada331ddc4`.

## Multi-speaker workloads

Generate all lines for one speaker while its reference embedding and prepared
prefix remain resident, then advance to the next speaker. Assemble the lines
back into script order after generation. This keeps the optimized decoder and
reference-side caches hot without changing the delivered dialogue order.

## Qualification record

The current native qualification used the original WordVoice checkpoints,
identical eight-line text, identical four reference voices, identical seeds,
and identical 0.45-second pauses. On the RTX 3060 it produced 46.31 seconds
of audio in 23.247533 seconds: RTF `0.501998`, or `1.992039x` real time.

Both gates passed:

- decoder eager parity: passed
- BF16 flow parity: passed (`max_relative_rmse=0.019265`,
  `min_cosine_similarity=0.99983066`)

Keep the JSON receipt with the generated audio. It is the evidence for the
engine hash, checkpoint hashes, line text hashes, reference hashes, and timing.
