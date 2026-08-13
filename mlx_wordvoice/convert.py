"""Convert the pinned WordVoice and MLX CosyVoice3 checkpoints into one model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
from safetensors.numpy import load_file, save_file


WORDVOICE_MODEL_REVISION = "bcde26261e0997329bb8f253b5c55a37ca1e5fa0"
WORDVOICE_SOURCE_REVISION = "fe9c9b7aa13093618627de0693d7974ff0981def"
MLX_AUDIO_PLUS_REVISION = "4c9ec6a8489e790b5ba8964ab1f1d63150476f9f"
MLX_BASE_REVISION = "18ccb7fbd7246e8cd3420d02f5dd28595cc0fcd9"
WORDVOICE_LLM_SHA256 = "c304f5636ab1ccbe383852c7815d51b3c32dc5ac03125ef8d6fcc54b84e6cf6b"
WORDVOICE_FLOW_SHA256 = "b644f444ae551b938e82e1dd38fa7d2e1925715e297b5ccd0b232e138a6035ad"
PYTORCH_RAND_NOISE_SEED = 0
PYTORCH_RAND_NOISE_SHAPE = (1, 80, 50 * 300)
PYTORCH_RAND_NOISE_VALUES_SHA256 = (
    "656c9256457b71d1621f32d64715e922c656670185e70b457f7734e2c4da0b95"
)

LLM_CONTROL_PREFIXES = (
    "boundary_embedding.",
    "boundary_predictor.",
    "duration_embedding.",
    "duration_predictor.",
    "energy_embedding.",
    "energy_predictor.",
    "f0_embedding.",
    "f0_predictor.",
    "pause_embedding.",
    "pause_predictor.",
    "tone_embedding.",
    "tone_predictor.",
)
FLOW_CONTROL_PREFIXES = (
    "bnd_embed.",
    "control_modulator.",
    "energy_embed.",
    "f0_embed.",
    "tone_embed.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    # np.swapaxes returns a strided view. safetensors 0.8 serializes that view
    # using its underlying contiguous byte order while retaining the swapped
    # shape, which silently scrambles every converted Conv1d kernel. Materialize
    # the MLX (out, kernel, in) layout before it reaches save_file.
    return np.ascontiguousarray(np.swapaxes(weight, 1, 2))


def convert_llm_state(
    state: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return consolidated base-compatible and WordVoice-only LLM weights."""
    base: dict[str, np.ndarray] = {}
    controls: dict[str, np.ndarray] = {}
    qwen_prefix = "llm.model.model."
    for key, value in state.items():
        if key.startswith(qwen_prefix):
            base["qwen2.model." + key[len(qwen_prefix) :]] = value
        elif key.startswith(("speech_embedding.", "llm_decoder.")):
            base["llm." + key] = value
        elif key.startswith(LLM_CONTROL_PREFIXES):
            controls["wordvoice_llm." + key] = value
    return base, controls


def convert_flow_state(
    state: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return consolidated base-compatible and WordVoice-only flow weights."""
    base: dict[str, np.ndarray] = {}
    controls: dict[str, np.ndarray] = {}
    for original_key, original_value in state.items():
        key = original_key
        value = original_value
        if key.startswith(FLOW_CONTROL_PREFIXES):
            if key.startswith("control_modulator."):
                key = key.replace("control_modulator.", "control_modulator.layers.", 1)
            controls["wordvoice_flow." + key] = value
            continue
        if "decoder.estimator.time_embed." in key:
            key = key.replace("time_embed.time_mlp.0", "time_embed.time_mlp_0")
            key = key.replace("time_embed.time_mlp.2", "time_embed.time_mlp_2")
        if "conv_pos_embed" in key:
            if ".conv1.0." in key:
                key = key.replace(".conv1.0.", ".conv1.")
                if key.endswith("weight") and value.ndim == 3:
                    value = transpose_conv_weight(value)
            elif ".conv2.0." in key:
                key = key.replace(".conv2.0.", ".conv2.")
                if key.endswith("weight") and value.ndim == 3:
                    value = transpose_conv_weight(value)
        if "transformer_blocks." in key:
            key = key.replace(".to_out.0.", ".to_out_0.")
            key = key.replace(".to_out.1.", ".to_out_1.")
            key = key.replace(".ff.ff.0.0.", ".ff.ff_0_0.")
            key = key.replace(".ff.ff.1.", ".ff.ff_1.")
            key = key.replace(".ff.ff.2.", ".ff.ff_2.")
        if "pre_lookahead_layer." in key and (
            key.endswith("conv1.weight") or key.endswith("conv2.weight")
        ) and value.ndim == 3:
            value = transpose_conv_weight(value)
        base["flow." + key] = value
    return base, controls


def _torch_state(path: Path) -> dict[str, np.ndarray]:
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"checkpoint must contain a state dictionary: {path}")
    result = {}
    for key, value in loaded.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            if value.is_floating_point():
                value = value.to(torch.float16)
            result[str(key)] = value.numpy()
    return result


def build_pytorch_rand_noise() -> np.ndarray:
    """Reproduce the fixed noise created by PyTorch CausalConditionalCFM."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(PYTORCH_RAND_NOISE_SEED)
    noise = torch.randn(
        PYTORCH_RAND_NOISE_SHAPE,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    ).numpy()
    actual = hashlib.sha256(noise.tobytes()).hexdigest()
    if actual != PYTORCH_RAND_NOISE_VALUES_SHA256:
        raise RuntimeError(
            "PyTorch fixed diffusion noise does not match the admitted values: "
            f"expected {PYTORCH_RAND_NOISE_VALUES_SHA256}, actual {actual}"
        )
    return noise


def _verify_file(path: Path, expected_sha256: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, actual {actual}, path {path}"
        )


def _quantize_qwen_weights(
    weights: Mapping[str, object], *, bits: int, group_size: int
) -> dict[str, object]:
    """Quantize only Qwen transformer Linear layers for faster autoregression."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model as Qwen2Model
    from mlx_lm.models.qwen2 import ModelArgs

    if bits not in {4, 8}:
        raise ValueError(f"Qwen quantization bits must be 4 or 8, actual={bits}")
    if group_size != 64:
        raise ValueError(
            f"Qwen quantization group size must be 64, actual={group_size}"
        )
    model = Qwen2Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=896,
            intermediate_size=4864,
            num_attention_heads=14,
            num_hidden_layers=24,
            num_key_value_heads=2,
            vocab_size=151936,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            tie_word_embeddings=True,
        )
    )
    source = {
        key[len("qwen2.") :]: value
        for key, value in weights.items()
        if key.startswith("qwen2.") and key != "qwen2.lm_head.weight"
    }
    model.load_weights(list(source.items()))
    mx.eval(model.parameters())

    def should_quantize(path, module):
        return isinstance(module, nn.Linear) and "model.layers" in path

    nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        class_predicate=should_quantize,
    )
    mx.eval(model.parameters())
    return {
        "qwen2." + key: value for key, value in tree_flatten(model.parameters())
    }


def quantize_existing_model(
    *,
    source_model: Path,
    destination: Path,
    bits: int,
    group_size: int = 64,
) -> dict[str, object]:
    """Create an immutable selective-Qwen candidate from an admitted v3 model."""
    import mlx.core as mx

    from .model_manifest import MODEL_CONTRACT, validate_model_manifest

    source_model = source_model.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_manifest = validate_model_manifest(source_model)
    if source_manifest.get("contract") != MODEL_CONTRACT:
        raise ValueError(
            "selective Qwen quantization requires an admitted FP16 v3 source; "
            f"actual={source_manifest.get('contract')!r}"
        )
    source_weights_path = source_model / "model.safetensors"
    source_files = source_manifest.get("files")
    source_weights_manifest = (
        source_files.get("model.safetensors")
        if isinstance(source_files, dict)
        else None
    )
    if not isinstance(source_weights_manifest, dict):
        raise ValueError("source v3 manifest is missing model.safetensors metadata")
    _verify_file(
        source_weights_path,
        str(source_weights_manifest.get("sha256")),
        "source MLX WordVoice model",
    )

    weights = mx.load(str(source_weights_path))
    quantized_qwen = _quantize_qwen_weights(
        weights, bits=bits, group_size=group_size
    )
    retained = {
        key: value for key, value in weights.items() if not key.startswith("qwen2.")
    }
    candidate = {**retained, **quantized_qwen}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        model_path = temporary / "model.safetensors"
        mx.save_safetensors(
            str(model_path),
            candidate,
            metadata={
                "format": "wordvoice-mlx-selective-qwen-v4",
                "qwen_bits": str(bits),
                "qwen_group_size": str(group_size),
                "source_model_sha256": str(source_weights_manifest["sha256"]),
            },
        )
        for source in source_model.iterdir():
            if source.name in {"model.safetensors", "wordvoice.json"} or not source.is_file():
                continue
            shutil.copy2(source, temporary / source.name)
        config_path = temporary / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["quantization"] = {
            "bits": bits,
            "group_size": group_size,
            "quantized_components": ["qwen2.model.layers"],
        }
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        noise_path = temporary / "rand_noise.npy"
        manifest = {
            "contract": "wordvoice-mlx-model.v4",
            "dtypes": {
                "qwen2_transformer_linear_weights": f"{bits}-bit-affine",
                "remaining_weights": "float16",
                "rand_noise": "float32",
            },
            "files": {
                "model.safetensors": {
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                },
                "rand_noise.npy": {
                    "bytes": noise_path.stat().st_size,
                    "sha256": sha256_file(noise_path),
                },
            },
            "inputs": source_manifest.get("inputs"),
            "parent": {
                "contract": MODEL_CONTRACT,
                "model_sha256": source_weights_manifest["sha256"],
            },
            "quantization": {
                "bits": bits,
                "components": ["qwen2.model.layers"],
                "group_size": group_size,
                "mode": "affine",
            },
            "tensor_count": len(candidate),
        }
        (temporary / "wordvoice.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def convert(
    *,
    base_model: Path,
    llm_checkpoint: Path,
    flow_checkpoint: Path,
    destination: Path,
) -> dict[str, object]:
    """Create an immutable FP16 MLX WordVoice model directory."""
    base_model = base_model.resolve()
    llm_checkpoint = llm_checkpoint.resolve()
    flow_checkpoint = flow_checkpoint.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    base_weights_path = base_model / "model.safetensors"
    if not base_weights_path.is_file():
        raise FileNotFoundError(f"MLX base model is missing model.safetensors: {base_model}")
    _verify_file(llm_checkpoint, WORDVOICE_LLM_SHA256, "WordVoice LLM checkpoint")
    _verify_file(flow_checkpoint, WORDVOICE_FLOW_SHA256, "WordVoice flow checkpoint")

    base_weights = load_file(base_weights_path)
    retained = {
        key: value
        for key, value in base_weights.items()
        if not key.startswith(("qwen2.", "llm.", "flow."))
    }
    llm_base, llm_controls = convert_llm_state(_torch_state(llm_checkpoint))
    flow_base, flow_controls = convert_flow_state(_torch_state(flow_checkpoint))
    rand_noise = build_pytorch_rand_noise()
    converted = {
        key: np.ascontiguousarray(value)
        for key, value in {
            **retained,
            **llm_base,
            **flow_base,
            **llm_controls,
            **flow_controls,
        }.items()
    }

    required_prefixes = (
        "qwen2.",
        "llm.speech_embedding.",
        "llm.llm_decoder.",
        "flow.decoder.",
        "wordvoice_llm.duration_embedding.",
        "wordvoice_flow.control_modulator.",
        "hifigan.",
    )
    missing = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in converted)]
    if missing:
        raise ValueError(f"converted checkpoint is missing required weight groups: {missing}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        save_file(
            converted,
            temporary / "model.safetensors",
            metadata={
                "format": "wordvoice-mlx-fp16-v3",
                "mlx_audio_plus_revision": MLX_AUDIO_PLUS_REVISION,
                "mlx_base_revision": MLX_BASE_REVISION,
                "wordvoice_model_revision": WORDVOICE_MODEL_REVISION,
                "wordvoice_source_revision": WORDVOICE_SOURCE_REVISION,
            },
        )
        for source in base_model.iterdir():
            if (
                source.name in {"model.safetensors", "rand_noise.npy"}
                or not source.is_file()
            ):
                continue
            shutil.copy2(source, temporary / source.name)
        rand_noise_path = temporary / "rand_noise.npy"
        np.save(rand_noise_path, rand_noise)
        model_path = temporary / "model.safetensors"
        manifest = {
            "contract": "wordvoice-mlx-model.v3",
            "dtypes": {
                "rand_noise": "float32",
                "weights": "float16",
            },
            "files": {
                "model.safetensors": {
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                },
                "rand_noise.npy": {
                    "bytes": rand_noise_path.stat().st_size,
                    "sha256": sha256_file(rand_noise_path),
                }
            },
            "inputs": {
                "mlx_audio_plus_revision": MLX_AUDIO_PLUS_REVISION,
                "mlx_base_model_revision": MLX_BASE_REVISION,
                "mlx_base_model_sha256": sha256_file(base_weights_path),
                "pytorch_rand_noise_seed": PYTORCH_RAND_NOISE_SEED,
                "pytorch_rand_noise_shape": list(PYTORCH_RAND_NOISE_SHAPE),
                "pytorch_rand_noise_values_sha256": (
                    PYTORCH_RAND_NOISE_VALUES_SHA256
                ),
                "wordvoice_flow_sha256": WORDVOICE_FLOW_SHA256,
                "wordvoice_llm_sha256": WORDVOICE_LLM_SHA256,
                "wordvoice_model_revision": WORDVOICE_MODEL_REVISION,
                "wordvoice_source_revision": WORDVOICE_SOURCE_REVISION,
            },
            "tensor_count": len(converted),
        }
        (temporary / "wordvoice.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--llm-checkpoint", type=Path)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument(
        "--source-model",
        type=Path,
        help="Admitted FP16 v3 model to selectively quantize",
    )
    parser.add_argument("--qwen-bits", type=int, choices=(4, 8))
    parser.add_argument("--qwen-group-size", type=int, default=64)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.source_model is not None:
        if any(
            value is not None
            for value in (args.base_model, args.llm_checkpoint, args.flow_checkpoint)
        ):
            parser.error(
                "--source-model cannot be combined with source checkpoint arguments"
            )
        if args.qwen_bits is None:
            parser.error("--source-model requires --qwen-bits")
        manifest = quantize_existing_model(
            source_model=args.source_model,
            destination=args.destination,
            bits=args.qwen_bits,
            group_size=args.qwen_group_size,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--base-model", args.base_model),
                ("--llm-checkpoint", args.llm_checkpoint),
                ("--flow-checkpoint", args.flow_checkpoint),
            )
            if value is None
        ]
        if missing:
            parser.error(f"source checkpoint conversion requires {', '.join(missing)}")
        if args.qwen_bits is not None:
            parser.error("--qwen-bits requires --source-model")
        manifest = convert(
            base_model=args.base_model,
            llm_checkpoint=args.llm_checkpoint,
            flow_checkpoint=args.flow_checkpoint,
            destination=args.destination,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
