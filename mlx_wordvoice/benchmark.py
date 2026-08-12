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
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    output.mkdir(parents=True)
    request = PreparedRequest.load(args.request)
    load_started = time.perf_counter()
    model = load_wordvoice_mlx(args.model, compile_flow=args.compile_flow)
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
        "contract": "wordvoice-mlx-qualification.v1",
        "load_seconds": round(load_seconds, 6),
        "model_manifest": json.loads(
            (args.model / "wordvoice.json").read_text(encoding="utf-8")
        ),
        "parity": {
            "prosody_control_sha256": next(iter(control_hashes)),
            "speech_token_count": next(iter(token_counts)),
            "speech_token_sha256": next(iter(token_hashes)),
            "status": "passed",
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
