"""Validate the immutable native MLX WordVoice model package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODEL_CONTRACT = "wordvoice-mlx-model.v3"
QUANTIZED_MODEL_CONTRACT = "wordvoice-mlx-model.v4"
SUPPORTED_MODEL_CONTRACTS = {MODEL_CONTRACT, QUANTIZED_MODEL_CONTRACT}


def validate_model_manifest(model_path: Path) -> dict[str, object]:
    manifest_path = model_path / "wordvoice.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract")
    if contract not in SUPPORTED_MODEL_CONTRACTS:
        if contract == "wordvoice-mlx-model.v1":
            reason = "its non-contiguous Conv1d kernels were serialized incorrectly"
        elif contract == "wordvoice-mlx-model.v2":
            reason = "it omits PyTorch's fixed diffusion-noise tensor"
        else:
            reason = "the contract is unknown"
        raise ValueError(
            f"unsupported MLX WordVoice model contract {contract!r}: {reason}; "
            "safe_recovery=reconvert-with-wordvoice-mlx-model.v3"
        )

    if contract == QUANTIZED_MODEL_CONTRACT:
        quantization = manifest.get("quantization")
        if not isinstance(quantization, dict):
            raise ValueError(
                "MLX WordVoice v4 manifest is missing quantization metadata; "
                "safe_recovery=rebuild-from-an-admitted-v3-model"
            )
        bits = quantization.get("bits")
        group_size = quantization.get("group_size")
        components = quantization.get("components")
        if bits not in {4, 8} or group_size != 64 or components != [
            "qwen2.model.layers"
        ]:
            raise ValueError(
                "MLX WordVoice v4 quantization contract mismatch: "
                f"bits={bits!r}, group_size={group_size!r}, "
                f"components={components!r}; "
                "safe_recovery=rebuild-with-selective-qwen-quantization"
            )

    files = manifest.get("files")
    noise_manifest = files.get("rand_noise.npy") if isinstance(files, dict) else None
    if not isinstance(noise_manifest, dict):
        raise ValueError(
            "MLX WordVoice v3 manifest is missing rand_noise.npy metadata; "
            "safe_recovery=reconvert-with-wordvoice-mlx-model.v3"
        )
    noise_path = model_path / "rand_noise.npy"
    if not noise_path.is_file():
        raise FileNotFoundError(
            "MLX WordVoice v3 is missing PyTorch's fixed diffusion-noise tensor: "
            f"{noise_path}; safe_recovery=reconvert-with-wordvoice-mlx-model.v3"
        )
    actual_noise_size = noise_path.stat().st_size
    expected_noise_size = noise_manifest.get("bytes")
    if actual_noise_size != expected_noise_size:
        raise ValueError(
            "MLX WordVoice rand_noise.npy size mismatch: "
            f"expected {expected_noise_size}, actual {actual_noise_size}, "
            f"path {noise_path}; safe_recovery=reconvert-with-wordvoice-mlx-model.v3"
        )
    noise_digest = hashlib.sha256(noise_path.read_bytes()).hexdigest()
    expected_noise_digest = noise_manifest.get("sha256")
    if noise_digest != expected_noise_digest:
        raise ValueError(
            "MLX WordVoice rand_noise.npy SHA-256 mismatch: "
            f"expected {expected_noise_digest}, actual {noise_digest}, "
            f"path {noise_path}; safe_recovery=reconvert-with-wordvoice-mlx-model.v3"
        )
    return manifest
