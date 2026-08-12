"""Resumable Storyforge narration queue for the local WordVoice runtime.

The queue owns only temporary chunk work and receipts.  It never changes the
voice catalog or the Storyforge manuscript.  Existing Chatterbox outputs are
left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

WORDVOICE_ROOT = Path(__file__).resolve().parents[1]
if str(WORDVOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORDVOICE_ROOT))

import wordvoice_infer as wv  # noqa: E402

RUNTIME_REVISION = wv.WORDVOICE_RUNTIME_REVISION
VOICE_ID = "DRAFT_VOICE_08"
REFERENCE_SHA256 = "50603e614e21668436e6be68e6b1bcd27b932c09285dd161a076117f2ef0de5b"
SAMPLE_RATE = 24000
MAX_ATTEMPTS = 3


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_aligner_directory(path: Path) -> Path:
    directory = path.parent if path.is_file() else path
    model_path = directory / "model.pt"
    if not directory.is_dir() or not model_path.is_file():
        raise FileNotFoundError(
            f"MMS-FA aligner must be a directory containing model.pt: {path}"
        )
    return directory


def resolve_runtime_identity(path: Path, require_qualified: bool) -> dict[str, Any]:
    engine_path = path / "decoder.plan"
    manifest_path = path / "manifest.json"
    if not engine_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"WordVoice TensorRT runtime must contain decoder.plan and manifest.json: {path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if require_qualified and status != "qualified":
        raise RuntimeError(
            f"WordVoice runtime is not qualified: {path} status={status!r}"
        )
    return {
        "directory": str(path),
        "engine_sha256": sha256_file(engine_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_status": status,
    }


def first_sentence(markdown: str) -> str:
    match = re.search(r"(?ms)^## Episode.*?\n\n(?:### .*?\n\n)?(.*?)\n\n", markdown)
    if not match:
        raise ValueError("episode manuscript has no first paragraph")
    paragraph = match.group(1).strip()
    sentence = re.split(r"(?<=[.!?])\s+", paragraph, maxsplit=1)[0]
    return sentence.strip()


def load_chunks(args: argparse.Namespace, episode: int) -> list[dict[str, Any]]:
    script_path: Path | None
    if episode == 1:
        script_path = args.episode_one_script
    else:
        script_path = args.script_root / f"episode-{episode:02d}" / "chatterbox-script.json"
    if script_path is not None and script_path.exists():
        source = json.loads(script_path.read_text(encoding="utf-8"))
        chunks: list[dict[str, Any]] = []
        for item in source["script"]:
            if "id" in item:
                chunks.append({
                    "id": item["id"],
                    "text": item["text"],
                    "display_text": item.get("display_text", item["text"]),
                    "pause_seconds": float(item.get("pause_seconds", 0.0)),
                })
            elif "pause" in item and chunks:
                chunks[-1]["pause_seconds"] = float(item["pause"])
        if chunks:
            return chunks

    plan = json.loads(args.plan.read_text(encoding="utf-8"))["chunks"]
    starts: list[int] = []
    for number in range(1, 9):
        text = (args.manuscript_root / f"episode-{number:02d}.md").read_text(encoding="utf-8")
        target = first_sentence(text)
        matches = [index for index, item in enumerate(plan) if item["text"] == target]
        if len(matches) != 1:
            raise ValueError(f"could not uniquely locate Episode {number} in speech plan: {target[:80]}")
        starts.append(matches[0])
    start = starts[episode - 1]
    end = starts[episode] if episode < 8 else len(plan)
    return [
        {
            "id": f"phrase-{index + 1:04d}",
            "text": item["text"],
            "display_text": item["text"],
            "pause_seconds": args.default_pause_seconds,
        }
        for index, item in enumerate(plan[start:end])
    ]


def format_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def initialize_runtime(args: argparse.Namespace) -> None:
    args.runtime_identity = resolve_runtime_identity(
        args.trt_runtime_dir, args.require_qualified_runtime
    )
    aligner_directory = resolve_aligner_directory(args.aligner_path)
    wv.Aligner_Model = wv.MMSFA_Aligner(model_path=str(aligner_directory))
    if args.text_frontend == "none":
        os.environ["COSYVOICE_DISABLE_TEXT_FRONTEND"] = "1"
    wv.wordvoice = wv.WordVoice(
        model_dir=str(args.cosyvoice_path),
        llm_path=str(args.llm_path),
        flow_path=str(args.flow_path),
        hyper_yaml_path=str(args.hyper_yaml_path),
        llm_trt_runtime=str(args.trt_runtime_dir),
        flow_amp_dtype="bf16",
    )


def synthesize_chunk(args: argparse.Namespace, chunk: dict[str, Any], wav_path: Path) -> dict[str, Any]:
    started = time.time()
    metrics = wv.eval(
        args.prompt_text,
        str(args.prompt_audio),
        chunk["text"],
        {},
        str(wav_path),
        "en",
        text_frontend=args.text_frontend == "auto",
    )
    info = sf.info(wav_path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
        raise RuntimeError(
            f"WordVoice output failed structural QC: {wav_path} "
            f"rate={info.samplerate} channels={info.channels} frames={info.frames}"
        )
    return {
        "status": "generated",
        "chunk_id": chunk["id"],
        "text": chunk["text"],
        "output": str(wav_path),
        "sha256": sha256_file(wav_path),
        "sample_rate_hz": info.samplerate,
        "channels": info.channels,
        "sample_count": info.frames,
        "duration_seconds": round(info.frames / info.samplerate, 6),
        "generation_seconds": round(time.time() - started, 3),
        "runtime_revision": RUNTIME_REVISION,
        "voice_id": VOICE_ID,
        "reference_sha256": REFERENCE_SHA256,
        "runtime_identity": args.runtime_identity,
        "runtime_metrics": metrics,
    }


def finalize_episode(args: argparse.Namespace, episode: int, chunks: list[dict[str, Any]], episode_dir: Path) -> None:
    final_path = episode_dir / f"episode-{episode:02d}.flac"
    srt_path = episode_dir / f"episode-{episode:02d}.srt"
    qc_path = episode_dir / "acoustic-qc.json"
    current = 0.0
    cues: list[tuple[float, float, str]] = []
    peak = 0.0
    squared = 0.0
    sample_count = 0
    with sf.SoundFile(final_path, mode="w", samplerate=SAMPLE_RATE, channels=1, format="FLAC", subtype="PCM_16") as output:
        for chunk in chunks:
            wav_path = episode_dir / "chunks" / f"{chunk['id']}.wav"
            audio, rate = sf.read(wav_path, dtype="float32")
            if rate != SAMPLE_RATE:
                raise RuntimeError(f"unexpected chunk rate for {wav_path}: {rate}")
            if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"chunk acoustic QC failed: {wav_path}")
            duration = len(audio) / rate
            output.write(audio)
            cues.append((current, current + duration, chunk["display_text"]))
            current += duration
            pause = max(0.0, float(chunk.get("pause_seconds", 0.0)))
            if pause:
                output.write(np.zeros(int(round(pause * rate)), dtype="float32"))
                current += pause
            peak = max(peak, float(np.max(np.abs(audio))))
            squared += float(np.sum(audio * audio))
            sample_count += len(audio)
    rms = float(np.sqrt(squared / max(sample_count, 1)))
    if peak <= 0.0 or rms <= 0.0:
        raise RuntimeError(f"episode acoustic QC failed: silent output {final_path}")
    with srt_path.open("w", encoding="utf-8", newline="\n") as srt:
        for index, (start, end, text) in enumerate(cues, start=1):
            srt.write(f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")
    atomic_json(qc_path, {
        "schema": "starline.wordvoice.acoustic-qc.v1",
        "status": "passed",
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "duration_seconds": round(current, 6),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
        "finite_samples": True,
        "no_time_stretch": True,
        "no_resampling": True,
        "no_pitch_processing": True,
    })
    atomic_json(episode_dir / "render-report.json", {
        "schema": "starline.wordvoice.render-report.v1",
        "status": "completed",
        "episode": episode,
        "voice_id": VOICE_ID,
        "reference_sha256": REFERENCE_SHA256,
        "runtime_revision": RUNTIME_REVISION,
        "runtime_identity": args.runtime_identity,
        "chunk_count": len(chunks),
        "audio": str(final_path),
        "audio_sha256": sha256_file(final_path),
        "subtitle": str(srt_path),
        "acoustic_qc": str(qc_path),
        "duration_seconds": round(current, 6),
    })


def run_episode(args: argparse.Namespace, episode: int) -> None:
    chunks = load_chunks(args, episode)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    episode_dir = args.output_root / f"episode-{episode:02d}"
    chunk_dir = episode_dir / "chunks"
    receipt_dir = episode_dir / "chunk-receipts"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(episode_dir / "prosody-plan.json", {
        "schema": "starline.wordvoice.prosody-plan.v1",
        "episode": episode,
        "voice_id": VOICE_ID,
        "reference_sha256": REFERENCE_SHA256,
        "runtime_revision": RUNTIME_REVISION,
        "runtime_identity": args.runtime_identity,
        "language": "en-US",
        "delivery": "Naturally unhurried, grounded, clear, low-energy conversational narration with ordinary sentence endings.",
        "chunks": [{"id": c["id"], "text": c["text"], "pause_seconds": c["pause_seconds"]} for c in chunks],
    })
    completed: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        wav_path = chunk_dir / f"{chunk['id']}.wav"
        receipt_path = receipt_dir / f"{chunk['id']}.json"
        if receipt_path.exists() and wav_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") == "generated" and receipt.get("sha256") == sha256_file(wav_path):
                completed.append(receipt)
                print(f"[resume] episode={episode:02d} chunk={index}/{len(chunks)} id={chunk['id']}", flush=True)
                continue
        last_error: str | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                receipt = synthesize_chunk(args, chunk, wav_path)
                receipt["attempt"] = attempt
                atomic_json(receipt_path, receipt)
                completed.append(receipt)
                print(f"[done] episode={episode:02d} chunk={index}/{len(chunks)} id={chunk['id']} duration={receipt['duration_seconds']:.2f}s", flush=True)
                break
            except Exception as error:  # noqa: BLE001
                last_error = repr(error)
                atomic_json(receipt_path, {
                    "status": "failed_attempt",
                    "episode": episode,
                    "chunk_id": chunk["id"],
                    "attempt": attempt,
                    "error": last_error,
                    "runtime_revision": RUNTIME_REVISION,
                })
                print(f"[retry] episode={episode:02d} chunk={index}/{len(chunks)} attempt={attempt} error={last_error}", flush=True)
        else:
            raise RuntimeError(f"chunk exhausted retry budget: episode={episode} id={chunk['id']} error={last_error}")
    if len(completed) == len(chunks):
        finalize_episode(args, episode, chunks, episode_dir)
        atomic_json(episode_dir / "terminal.json", {
            "schema": "starline.wordvoice.terminal.v1",
            "status": "completed",
            "episode": episode,
            "voice_id": VOICE_ID,
            "reference_sha256": REFERENCE_SHA256,
            "runtime_revision": RUNTIME_REVISION,
            "chunk_count": len(chunks),
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-owner", required=True)
    parser.add_argument("--agent-task", required=True)
    parser.add_argument("--episode", type=int, nargs="+", required=True)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument("--script-root", type=Path, required=True)
    parser.add_argument("--episode-one-script", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--cosyvoice-path", type=Path, required=True)
    parser.add_argument("--aligner-path", type=Path, required=True)
    parser.add_argument("--llm-path", type=Path, required=True)
    parser.add_argument("--flow-path", type=Path, required=True)
    parser.add_argument("--hyper-yaml-path", type=Path, required=True)
    parser.add_argument("--trt-runtime-dir", type=Path, required=True)
    parser.add_argument("--require-qualified-runtime", action="store_true")
    parser.add_argument("--text-frontend", choices=("none", "auto"), default="none")
    parser.add_argument("--default-pause-seconds", type=float, default=0.707)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_runtime(args)
    for episode in args.episode:
        run_episode(args, episode)


if __name__ == "__main__":
    main()
