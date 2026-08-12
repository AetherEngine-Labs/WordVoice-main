"""Load and execute the converted native MLX WordVoice model."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.tts.models.cosyvoice3 import CosyVoice3, load_cosyvoice3

from .contract import PreparedRequest
from .flow import WordVoiceFlow
from .llm import WordVoiceLM


@dataclass(frozen=True)
class SynthesisResult:
    audio: mx.array
    sample_rate: int
    speech_tokens: tuple[int, ...]
    controls: dict[str, tuple[int, ...]]
    metrics: dict[str, object]


class WordVoiceMLX(CosyVoice3):
    def synthesize_prepared(
        self,
        request: PreparedRequest,
        *,
        seed: int,
        use_prepared_prefix: bool = True,
    ) -> SynthesisResult:
        request.validate()
        mx.random.seed(seed)
        controls = request.controls
        started = time.perf_counter()
        token_result = self.llm.infer_controlled(
            key=request.fingerprint(),
            text=mx.array(request.text_tokens),
            prompt_text=mx.array(request.prompt_text_tokens),
            prompt_speech_token=mx.array(request.prompt_speech_tokens),
            word_tokens=tuple(mx.array(value) for value in request.word_tokens),
            starts=controls.starts,
            durations=controls.durations,
            boundaries=controls.boundaries,
            tones=controls.tones,
            pitches=controls.pitches,
            energies=controls.energies,
            use_prepared_prefix=use_prepared_prefix,
        )
        mx.synchronize()
        llm_seconds = time.perf_counter() - started
        token_array = mx.array([token_result.speech_tokens], dtype=mx.int32)
        flow_started = time.perf_counter()
        mel = self.flow.inference_wordvoice(
            token=token_array,
            token_len=mx.array([len(token_result.speech_tokens)], dtype=mx.int32),
            prompt_token=mx.array(request.prompt_speech_tokens),
            prompt_token_len=mx.array(
                [request.prompt_speech_tokens.shape[1]], dtype=mx.int32
            ),
            start_id=controls.starts[0],
            durations=token_result.durations,
            boundaries=token_result.boundaries,
            tones=token_result.tones,
            pitches=token_result.pitches,
            energies=token_result.energies,
            prompt_feat=mx.array(request.prompt_mel),
            prompt_feat_len=mx.array([request.prompt_mel.shape[1]], dtype=mx.int32),
            embedding=mx.array(request.speaker_embedding),
        )
        mx.synchronize()
        flow_seconds = time.perf_counter() - flow_started
        vocoder_started = time.perf_counter()
        audio = self.mel_to_audio(mel, finalize=True)
        mx.eval(audio)
        mx.synchronize()
        vocoder_seconds = time.perf_counter() - vocoder_started
        token_hash = hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little", signed=True) for token in token_result.speech_tokens)
        ).hexdigest()
        plan = {
            "boundaries": token_result.boundaries,
            "durations": token_result.durations,
            "energies": token_result.energies,
            "pauses": token_result.pauses,
            "pitches": token_result.pitches,
            "tones": token_result.tones,
        }
        control_hash = hashlib.sha256(
            json.dumps(plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        target_frames = int(mel.shape[2])
        audio_samples = int(audio.shape[-1])
        if audio_samples <= 0:
            raise RuntimeError(
                "WordVoice vocoder produced no audio samples; "
                "safe_recovery=preserve-request-and-diagnose-vocoder-output"
            )
        duration_seconds = audio_samples / self.sample_rate
        audio_float = audio.astype(mx.float32)
        audio_peak = float(mx.max(mx.abs(audio_float)).item())
        audio_rms = float(mx.sqrt(mx.mean(audio_float * audio_float)).item())
        audio_peak_dbfs = 20.0 * math.log10(max(audio_peak, 1e-12))
        audio_rms_dbfs = 20.0 * math.log10(max(audio_rms, 1e-12))
        metrics = {
            **token_result.metrics,
            "audio_peak_dbfs": round(audio_peak_dbfs, 6),
            "audio_rms_dbfs": round(audio_rms_dbfs, 6),
            "audio_seconds": round(duration_seconds, 6),
            "flow_seconds": round(flow_seconds, 6),
            "flow_runtime": "compiled-step-v1"
            if self.flow.compilation_enabled
            else "eager",
            "llm_seconds": round(llm_seconds, 6),
            "model_seconds": round(llm_seconds + flow_seconds + vocoder_seconds, 6),
            "prosody_control_sha256": control_hash,
            "real_time_factor": round(
                (llm_seconds + flow_seconds + vocoder_seconds) / duration_seconds, 6
            ),
            "seed": seed,
            "speech_token_count": len(token_result.speech_tokens),
            "speech_token_sha256": token_hash,
            "target_mel_frames": target_frames,
            "vocoder_seconds": round(vocoder_seconds, 6),
        }
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            speech_tokens=token_result.speech_tokens,
            controls=plan,
            metrics=metrics,
        )


def load_wordvoice_mlx(
    model_path: Path, *, compile_flow: bool = False
) -> WordVoiceMLX:
    model_path = model_path.resolve()
    manifest = json.loads((model_path / "wordvoice.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != "wordvoice-mlx-model.v2":
        raise ValueError(
            "unsupported MLX WordVoice model contract; the v1 conversion is rejected "
            "because non-contiguous Conv1d views were serialized with scrambled kernels"
        )
    base = load_cosyvoice3(str(model_path), dtype=mx.float16)
    all_weights = mx.load(str(model_path / "model.safetensors"))
    llm = WordVoiceLM(llm=base.llm.llm)
    llm_weights = {
        key[4:]: value for key, value in all_weights.items() if key.startswith("llm.")
    }
    llm_weights.update(
        {
            key[len("wordvoice_llm.") :]: value
            for key, value in all_weights.items()
            if key.startswith("wordvoice_llm.")
        }
    )
    expected_llm_weights = {
        key
        for key, _ in tree_flatten(llm.parameters())
        if not key.startswith("llm.")
    }
    actual_llm_weights = set(llm_weights)
    if actual_llm_weights != expected_llm_weights:
        raise ValueError(
            "WordVoice-owned LLM weight mismatch: "
            f"missing={sorted(expected_llm_weights - actual_llm_weights)}, "
            f"unexpected={sorted(actual_llm_weights - expected_llm_weights)}"
        )
    # Qwen is the exact model already loaded by load_cosyvoice3 above. Load only
    # the WordVoice-owned embedding, projection, control, and prediction layers.
    llm.load_weights(list(llm_weights.items()), strict=False)
    flow = WordVoiceFlow(
        input_size=base.flow.input_size,
        output_size=base.flow.output_size,
        spk_embed_dim=192,
        vocab_size=base.flow.vocab_size,
        input_frame_rate=base.flow.input_frame_rate,
        token_mel_ratio=base.flow.token_mel_ratio,
        pre_lookahead_len=base.flow.pre_lookahead_len,
        pre_lookahead_layer=base.flow.pre_lookahead_layer,
        decoder=base.flow.decoder,
        n_timesteps=base.flow.n_timesteps,
    )
    flow_weights = {
        key[5:]: value
        for key, value in all_weights.items()
        if key.startswith("flow.") and "rotary_embed.inv_freq" not in key
    }
    flow_weights.update(
        {
            key[len("wordvoice_flow.") :]: value
            for key, value in all_weights.items()
            if key.startswith("wordvoice_flow.")
        }
    )
    flow.load_weights(list(flow_weights.items()), strict=True)
    model = WordVoiceMLX(config=base.config, llm=llm, flow=flow, hifigan=base.hifigan)
    model.eval()
    mx.eval(model.parameters())
    if compile_flow:
        model.flow.enable_compilation()
    return model
