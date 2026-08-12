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
    return np.swapaxes(weight, 1, 2)


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


def _verify_file(path: Path, expected_sha256: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, actual {actual}, path {path}"
        )


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
    converted = {**retained, **llm_base, **flow_base, **llm_controls, **flow_controls}

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
                "format": "wordvoice-mlx-fp16-v1",
                "mlx_audio_plus_revision": MLX_AUDIO_PLUS_REVISION,
                "mlx_base_revision": MLX_BASE_REVISION,
                "wordvoice_model_revision": WORDVOICE_MODEL_REVISION,
                "wordvoice_source_revision": WORDVOICE_SOURCE_REVISION,
            },
        )
        for source in base_model.iterdir():
            if source.name == "model.safetensors" or not source.is_file():
                continue
            shutil.copy2(source, temporary / source.name)
        model_path = temporary / "model.safetensors"
        manifest = {
            "contract": "wordvoice-mlx-model.v1",
            "dtype": "float16",
            "files": {
                "model.safetensors": {
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                }
            },
            "inputs": {
                "mlx_audio_plus_revision": MLX_AUDIO_PLUS_REVISION,
                "mlx_base_model_revision": MLX_BASE_REVISION,
                "mlx_base_model_sha256": sha256_file(base_weights_path),
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
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--llm-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
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
