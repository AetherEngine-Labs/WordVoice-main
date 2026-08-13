# Native MLX WordVoice runtime

The `mlx_wordvoice` package ports the WordVoice language model, five explicit
word-level controls, flow decoder, and repeated-request prefix reuse to native
MLX. It is an explicit runtime choice; it never falls back to PyTorch or a
different voice engine.

The admitted Windows WordVoice frontend remains the owner of text
normalization, prompt alignment, reference analysis, and control construction.
It exports those exact tensors as a fingerprinted prepared-request bundle. The
Mac runtime consumes that bundle without recalculating or weakening the
prosody contract.

## Install

Install the pinned MLX dependencies on Apple Silicon:

```bash
python3 -m venv ~/.ainotebook/wordvoice-mlx-venv
~/.ainotebook/wordvoice-mlx-venv/bin/python -m pip install -e '.[mlx]'
```

## Convert the model

Conversion verifies the exact WordVoice checkpoint hashes and combines them
with the pinned MLX CosyVoice3 base. The destination must not already exist.

```bash
AGENT_OWNER=operator AGENT_TASK=wordvoice-mlx-convert \
~/.ainotebook/wordvoice-mlx-venv/bin/python \
  -X agent.owner=operator -X agent.task=wordvoice-mlx-convert \
  -m mlx_wordvoice.convert \
  --base-model /absolute/path/to/mlx-cosyvoice3-base \
  --llm-checkpoint /absolute/path/to/wordvoice_llm_en.pt \
  --flow-checkpoint /absolute/path/to/wordvoice_fm.pt \
  --destination /absolute/path/to/wordvoice-mlx-fp16-v3
```

The generated `wordvoice.json` records every source revision, checkpoint hash,
the tensor count, model byte count, and converted model SHA-256.

The v1 conversion is rejected. Its non-contiguous NumPy Conv1d views were
serialized in the wrong byte order, scrambling the pre-lookahead and DiT
convolution kernels even though their tensor names and shapes looked valid.
The v2 conversion repaired those kernels but omitted the fixed diffusion-noise
tensor created by PyTorch's `CausalConditionalCFM`. MLX then generated a
different tensor with its own random-number generator, so nearly identical
conditioning still produced a measurably different final mel. The v3 converter
materializes every tensor in contiguous MLX layout, reproduces and records the
exact PyTorch noise tensor, and refuses both superseded contracts.

## Export an exact request

Run `mlx_wordvoice.export_request_cli` from the admitted PyTorch environment.
It writes `request.json` plus `arrays.npz`; the fingerprint covers every input
tensor, word token, control vector, model identity, and reference-audio hash.

```powershell
$env:AGENT_OWNER='operator'
$env:AGENT_TASK='wordvoice-mlx-export'
python -X agent.owner=operator -X agent.task=wordvoice-mlx-export `
  -m mlx_wordvoice.export_request_cli `
  --cosyvoice-model C:\absolute\base-model `
  --aligner-model C:\absolute\mms-fa `
  --llm-checkpoint C:\absolute\wordvoice_llm_en.pt `
  --flow-checkpoint C:\absolute\wordvoice_fm.pt `
  --hyper-yaml C:\absolute\wordvoice.yaml `
  --prompt-audio C:\absolute\reference.wav `
  --prompt-text 'Exact reference transcript.' `
  --text 'Controlled narration text.' `
  --output C:\absolute\prepared-request
```

## Qualify prefix reuse and compiled flow

The benchmark runs cache-disabled, cache-miss, and cache-hit variants in
sequence. It fails unless every run produces the same speech-token count,
speech-token SHA-256, and final prosody-control SHA-256. `--compile-flow`
compiles the immutable pre-lookahead, control modulator, and individual Euler
decoder step while retaining explicit evaluation between ODE steps.

```bash
AGENT_OWNER=operator AGENT_TASK=wordvoice-mlx-qualify \
~/.ainotebook/wordvoice-mlx-venv/bin/python \
  -X agent.owner=operator -X agent.task=wordvoice-mlx-qualify \
  -m mlx_wordvoice.benchmark \
  --model /absolute/path/to/wordvoice-mlx-fp16-v3 \
  --request /absolute/path/to/prepared-request \
  --output /absolute/path/to/qualification \
  --seed 4244 \
  --compile-flow
```

The resulting `qualification.json` is the durable cache-parity and performance
receipt. It also rejects technically inactive audio below the declared peak and
RMS floors. Production admission still requires a representative listening
test; successful token/control and activity gates are not voice-quality
approval.

Eight Euler flow steps remain the evaluation default. On a fixed PyTorch
token/control plan, eight steps reduced same-process flow time by 25.4% and
retained a ten-step mel cosine of 0.999929. This comparison proved acoustic
parity between step counts, not natural speech. On 2026-08-13 Alexander
rejected both the FP16 eight-step audition and the selective-Qwen audition as
weird. The 108-word target had been compressed into about 27 seconds and its
stored ASR check contained unrelated fragments at the ending. No native MLX
WordVoice profile is production-approved. Omit `--flow-steps` only for
evaluation; use explicit `--flow-steps 10` for comparison or diagnosis.

## Build an explicit selective-Qwen candidate

The autoregressive Qwen transformer can be quantized without changing the
full-precision WordVoice controls, flow decoder, or vocoder. The converter
accepts only an admitted v3 package and writes a new immutable v4 destination;
it never modifies or replaces the FP16 evaluation model.

```bash
AGENT_OWNER=operator AGENT_TASK=wordvoice-mlx-quantize \
~/.ainotebook/wordvoice-mlx-venv/bin/python \
  -X agent.owner=operator -X agent.task=wordvoice-mlx-quantize \
  -m mlx_wordvoice.convert \
  --source-model /absolute/path/to/wordvoice-mlx-fp16-v3 \
  --qwen-bits 4 \
  --destination /absolute/path/to/wordvoice-mlx-qwen4-v4
```

Four- and eight-bit candidates are evaluation models until their sequential
benchmark receipts and representative audio receive explicit listening
approval. Quantization is never an automatic fallback from the FP16 runtime.
