"""Portable, deterministic WordVoice request and control contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MAX_DURATION = 35
MAX_BOUNDARY = 5
MAX_TONE = 7
MAX_PITCH = 20
MAX_ENERGY = 20


def _integers(name: str, values: Sequence[int], upper: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 or value > upper for value in result):
        raise ValueError(f"{name} values must be between 0 and {upper}")
    return result


@dataclass(frozen=True)
class ControlPlan:
    """Word-level controls for the complete prompt plus target sequence."""

    starts: tuple[int, ...]
    durations: tuple[int, ...]
    boundaries: tuple[int, ...]
    tones: tuple[int, ...]
    pitches: tuple[int, ...]
    energies: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        *,
        starts: Sequence[int],
        durations: Sequence[int],
        boundaries: Sequence[int],
        tones: Sequence[int],
        pitches: Sequence[int],
        energies: Sequence[int],
    ) -> "ControlPlan":
        plan = cls(
            starts=_integers("starts", starts, 1_000_000),
            durations=_integers("durations", durations, MAX_DURATION),
            boundaries=_integers("boundaries", boundaries, MAX_BOUNDARY),
            tones=_integers("tones", tones, MAX_TONE),
            pitches=_integers("pitches", pitches, MAX_PITCH),
            energies=_integers("energies", energies, MAX_ENERGY),
        )
        lengths = {
            len(plan.durations),
            len(plan.boundaries),
            len(plan.tones),
            len(plan.pitches),
            len(plan.energies),
        }
        if len(lengths) != 1 or not plan.durations:
            raise ValueError("all control vectors must have one identical non-zero length")
        if not plan.starts:
            raise ValueError("the prompt must contain at least one aligned word start")
        if tuple(sorted(plan.starts)) != plan.starts:
            raise ValueError("prompt word starts must be monotonically increasing")
        if len(plan.starts) > len(plan.durations):
            raise ValueError("prompt word starts cannot outnumber complete controls")
        return plan

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "boundaries": self.boundaries,
                "durations": self.durations,
                "energies": self.energies,
                "pitches": self.pitches,
                "starts": self.starts,
                "tones": self.tones,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


@dataclass(frozen=True)
class PreparedRequest:
    """Exact cross-runtime inputs produced by the admitted Windows frontend."""

    text_tokens: np.ndarray
    prompt_text_tokens: np.ndarray
    prompt_speech_tokens: np.ndarray
    prompt_mel: np.ndarray
    speaker_embedding: np.ndarray
    word_tokens: tuple[np.ndarray, ...]
    controls: ControlPlan
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        arrays = {
            "text_tokens": self.text_tokens,
            "prompt_text_tokens": self.prompt_text_tokens,
            "prompt_speech_tokens": self.prompt_speech_tokens,
            "prompt_mel": self.prompt_mel,
            "speaker_embedding": self.speaker_embedding,
        }
        for name, value in arrays.items():
            if not isinstance(value, np.ndarray) or value.size == 0:
                raise ValueError(f"{name} must be a non-empty NumPy array")
        if self.text_tokens.ndim != 2 or self.text_tokens.shape[0] != 1:
            raise ValueError("text_tokens must have shape (1, tokens)")
        if self.prompt_text_tokens.ndim != 2 or self.prompt_text_tokens.shape[0] != 1:
            raise ValueError("prompt_text_tokens must have shape (1, tokens)")
        if self.prompt_speech_tokens.ndim != 2 or self.prompt_speech_tokens.shape[0] != 1:
            raise ValueError("prompt_speech_tokens must have shape (1, tokens)")
        if self.prompt_mel.ndim != 3 or self.prompt_mel.shape[0] != 1:
            raise ValueError("prompt_mel must have shape (1, frames, channels)")
        if self.speaker_embedding.ndim != 2 or self.speaker_embedding.shape[0] != 1:
            raise ValueError("speaker_embedding must have shape (1, channels)")
        if len(self.word_tokens) != len(self.controls.durations):
            raise ValueError("word token count must equal control count")
        if any(word.ndim != 2 or word.shape[0] != 1 or word.size == 0 for word in self.word_tokens):
            raise ValueError("each word token array must have shape (1, tokens)")
        required = {
            "base_model_revision",
            "reference_audio_sha256",
            "wordvoice_model_revision",
            "wordvoice_source_revision",
        }
        missing = sorted(required.difference(self.metadata))
        if missing:
            raise ValueError(f"prepared request metadata is missing {missing}")

    def fingerprint(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        for name, value in (
            ("text_tokens", self.text_tokens),
            ("prompt_text_tokens", self.prompt_text_tokens),
            ("prompt_speech_tokens", self.prompt_speech_tokens),
            ("prompt_mel", self.prompt_mel),
            ("speaker_embedding", self.speaker_embedding),
        ):
            contiguous = np.ascontiguousarray(value)
            digest.update(name.encode("ascii"))
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        for index, word in enumerate(self.word_tokens):
            contiguous = np.ascontiguousarray(word)
            digest.update(f"word_{index}".encode("ascii"))
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        digest.update(self.controls.canonical_bytes())
        digest.update(
            json.dumps(dict(self.metadata), separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        )
        return digest.hexdigest()

    def save(self, destination: Path) -> None:
        self.validate()
        destination = destination.resolve()
        if destination.exists():
            raise FileExistsError(f"prepared request destination already exists: {destination}")
        destination.mkdir(parents=True)
        arrays = {
            "text_tokens": self.text_tokens,
            "prompt_text_tokens": self.prompt_text_tokens,
            "prompt_speech_tokens": self.prompt_speech_tokens,
            "prompt_mel": self.prompt_mel,
            "speaker_embedding": self.speaker_embedding,
        }
        for index, word in enumerate(self.word_tokens):
            arrays[f"word_{index:04d}"] = word
        np.savez(destination / "arrays.npz", **arrays)
        manifest = {
            "contract": "wordvoice-mlx-prepared-request.v1",
            "controls": json.loads(self.controls.canonical_bytes()),
            "fingerprint": self.fingerprint(),
            "metadata": dict(self.metadata),
            "word_count": len(self.word_tokens),
        }
        (destination / "request.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, source: Path) -> "PreparedRequest":
        source = source.resolve()
        manifest = json.loads((source / "request.json").read_text(encoding="utf-8"))
        if manifest.get("contract") != "wordvoice-mlx-prepared-request.v1":
            raise ValueError("unsupported prepared request contract")
        with np.load(source / "arrays.npz", allow_pickle=False) as stored:
            word_count = int(manifest["word_count"])
            controls = ControlPlan.from_sequences(**manifest["controls"])
            request = cls(
                text_tokens=stored["text_tokens"],
                prompt_text_tokens=stored["prompt_text_tokens"],
                prompt_speech_tokens=stored["prompt_speech_tokens"],
                prompt_mel=stored["prompt_mel"],
                speaker_embedding=stored["speaker_embedding"],
                word_tokens=tuple(stored[f"word_{index:04d}"] for index in range(word_count)),
                controls=controls,
                metadata=manifest["metadata"],
            )
        actual = request.fingerprint()
        if actual != manifest.get("fingerprint"):
            raise ValueError(
                "prepared request fingerprint mismatch: "
                f"expected {manifest.get('fingerprint')}, actual {actual}"
            )
        return request
