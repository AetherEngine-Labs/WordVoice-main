"""Export the exact admitted PyTorch frontend tensors for native MLX inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import ControlPlan, PreparedRequest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "contiguous"):
        value = value.contiguous()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def export_prepared_request(
    *,
    model_input: Mapping[str, Any],
    prompt_audio: Path,
    wordvoice_source_revision: str,
    wordvoice_model_revision: str,
    base_model_revision: str,
    destination: Path,
    metadata: Mapping[str, Any] | None = None,
) -> PreparedRequest:
    """Persist all tensors and controls before the PyTorch model executes."""
    request = PreparedRequest(
        text_tokens=_numpy(model_input["text"]),
        prompt_text_tokens=_numpy(model_input["prompt_text"]),
        prompt_speech_tokens=_numpy(model_input["llm_prompt_speech_token"]),
        prompt_mel=_numpy(model_input["prompt_speech_feat"]),
        speaker_embedding=_numpy(model_input["llm_embedding"]),
        word_tokens=tuple(_numpy(value) for value in model_input["word_list"]),
        controls=ControlPlan.from_sequences(
            starts=model_input["start_list"],
            durations=model_input["dur_list"],
            boundaries=model_input["bnd_list"],
            tones=model_input["tone_list"],
            pitches=model_input["f0_list"],
            energies=model_input["eng_list"],
        ),
        metadata={
            **dict(metadata or {}),
            "base_model_revision": base_model_revision,
            "reference_audio_sha256": sha256_file(prompt_audio),
            "wordvoice_model_revision": wordvoice_model_revision,
            "wordvoice_source_revision": wordvoice_source_revision,
        },
    )
    request.save(destination)
    return request
