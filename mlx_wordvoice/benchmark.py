"""Run serial cache parity and performance qualification for MLX WordVoice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .contract import PreparedRequest
from .model import load_wordvoice_mlx


MIN_AUDIO_PEAK_DBFS = -18.0
MIN_AUDIO_RMS_DBFS = -40.0


def metric_float(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        raise TypeError(f"MLX WordVoice metric {name!r} must be numeric, actual={value!r}")
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4244)
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument(
        "--flow-steps",
        type=int,
        help="Explicit Euler step count; omit to retain the model's ten-step default",
    )
    args = parser.parse_args()
    if args.flow_steps is not None and args.flow_steps < 1:
        parser.error("--flow-steps must be at least 1")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    output.mkdir(parents=True)
    request = PreparedRequest.load(args.request)
    load_started = time.perf_counter()
    model = load_wordvoice_mlx(
        args.model,
        compile_flow=args.compile_flow,
        flow_steps=args.flow_steps,
    )
    load_seconds = time.perf_counter() - load_started
    runs = []
    for label, cache_enabled in (
        ("cache-disabled", False),
        ("cache-miss", True),
        ("cache-hit", True),
    ):
        result = model.synthesize_prepared(
            request,
            seed=args.seed,
            use_prepared_prefix=cache_enabled,
        )
        audio = np.asarray(result.audio).squeeze().astype(np.float32, copy=False)
        wav_path = output / f"{label}.wav"
        sf.write(wav_path, audio, result.sample_rate, subtype="PCM_16")
        run = {
            "label": label,
            "metrics": result.metrics,
            "wav": {
                "bytes": wav_path.stat().st_size,
                "path": wav_path.name,
                "sha256": sha256_file(wav_path),
            },
        }
        runs.append(run)
        print(json.dumps(run, sort_keys=True), flush=True)

        peak_dbfs = metric_float(result.metrics, "audio_peak_dbfs")
        rms_dbfs = metric_float(result.metrics, "audio_rms_dbfs")
        if peak_dbfs < MIN_AUDIO_PEAK_DBFS or rms_dbfs < MIN_AUDIO_RMS_DBFS:
            raise RuntimeError(
                "MLX WordVoice audio activity gate failed: "
                f"peak_dbfs={peak_dbfs:.3f} (minimum {MIN_AUDIO_PEAK_DBFS:.3f}), "
                f"rms_dbfs={rms_dbfs:.3f} (minimum {MIN_AUDIO_RMS_DBFS:.3f}); "
                "safe_recovery=preserve-output-and-diagnose-flow-before-listening"
            )

    token_hashes = {run["metrics"]["speech_token_sha256"] for run in runs}
    control_hashes = {run["metrics"]["prosody_control_sha256"] for run in runs}
    token_counts = {run["metrics"]["speech_token_count"] for run in runs}
    if len(token_hashes) != 1 or len(control_hashes) != 1 or len(token_counts) != 1:
        raise RuntimeError(
            "MLX WordVoice prepared-prefix parity failed: "
            f"speech_token_sha256={sorted(token_hashes)}, "
            f"prosody_control_sha256={sorted(control_hashes)}, "
            f"speech_token_count={sorted(token_counts)}"
        )
    receipt = {
        "contract": "wordvoice-mlx-qualification.v2",
        "load_seconds": round(load_seconds, 6),
        "model_manifest": json.loads(
            (args.model / "wordvoice.json").read_text(encoding="utf-8")
        ),
        "parity": {
            "audio_activity_gate": "passed",
            "prosody_control_sha256": next(iter(control_hashes)),
            "speech_token_count": next(iter(token_counts)),
            "speech_token_sha256": next(iter(token_hashes)),
            "status": "passed",
        },
        "production_admission": {
            "listening_approval": "required",
            "status": "awaiting-human-listening",
        },
        "request_fingerprint": request.fingerprint(),
        "runs": runs,
        "runtime": {
            "agent_owner": os.environ.get("AGENT_OWNER"),
            "agent_task": os.environ.get("AGENT_TASK"),
        },
        "seed": args.seed,
    }
    receipt_path = output / "qualification.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, receipt_path)
    print(json.dumps(receipt["parity"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
