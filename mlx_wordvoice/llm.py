"""WordVoice's controlled speech-token language model, ported natively to MLX."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from mlx_audio.tts.models.cosyvoice3.llm import Qwen2Encoder, ras_sampling


@dataclass(frozen=True)
class PreparedWordVoicePrefix:
    owner: object
    key: str
    hidden: mx.array
    cache_states: tuple[tuple[mx.array, mx.array], ...]
    word_embeddings: tuple[mx.array, ...]
    word_index: int
    preceding_duration: int
    durations: tuple[int, ...]
    boundaries: tuple[int, ...]
    tones: tuple[int, ...]
    pitches: tuple[int, ...]
    energies: tuple[int, ...]
    final_duration: int
    final_boundary: int
    prepare_seconds: float
    retained_bytes: int


@dataclass(frozen=True)
class ControlledTokenResult:
    speech_tokens: tuple[int, ...]
    durations: tuple[int, ...]
    boundaries: tuple[int, ...]
    tones: tuple[int, ...]
    pitches: tuple[int, ...]
    energies: tuple[int, ...]
    pauses: tuple[int, ...]
    metrics: dict[str, object]


class WordVoiceLM(nn.Module):
    """CosyVoice3 Qwen2 decoder with WordVoice's exact control state machine."""

    def __init__(
        self,
        *,
        llm: Qwen2Encoder,
        sampling: Callable = ras_sampling,
        llm_input_size: int = 896,
        llm_output_size: int = 896,
        speech_token_size: int = 6561,
        extended_vocab_size: int = 200,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        self.extended_vocab_size = extended_vocab_size
        self.sos = speech_token_size
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.bound_token = speech_token_size + 3
        self.llm = llm
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + extended_vocab_size, bias=False
        )
        self.speech_embedding = nn.Embedding(
            speech_token_size + extended_vocab_size, llm_input_size
        )
        self.max_duration = 35
        self.max_pause = 200
        self.max_boundary = 5
        self.max_tone = 7
        self.max_f0 = 20
        self.max_energy = 20
        self.silent_tokens = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
        self.max_pause_lens = (0, 1, 4, 10, 15)
        self.duration_embedding = nn.Embedding(self.max_duration + 1, llm_input_size)
        self.pause_embedding = nn.Embedding(self.max_pause + 1, llm_input_size)
        self.boundary_embedding = nn.Embedding(self.max_boundary + 1, llm_input_size)
        self.tone_embedding = nn.Embedding(self.max_tone + 1, llm_input_size)
        self.f0_embedding = nn.Embedding(self.max_f0 + 1, llm_input_size)
        self.energy_embedding = nn.Embedding(self.max_energy + 1, llm_input_size)
        self.duration_predictor = nn.Linear(llm_output_size, self.max_duration + 1)
        self.pause_predictor = nn.Linear(llm_output_size, self.max_pause + 1)
        self.boundary_predictor = nn.Linear(llm_output_size, self.max_boundary + 1)
        self.tone_predictor = nn.Linear(llm_output_size, self.max_tone + 1)
        self.f0_predictor = nn.Linear(llm_output_size, self.max_f0 + 1)
        self.energy_predictor = nn.Linear(llm_output_size, self.max_energy + 1)
        self.sampling = sampling
        self._prepared_prefix: PreparedWordVoicePrefix | None = None
        self._prepared_prefix_hits = 0
        self._prepared_prefix_misses = 0

    def _sample(
        self,
        scores: mx.array,
        decoded_tokens: list[int],
        sampling: int,
        *,
        ignore_eos: bool,
    ) -> int:
        if ignore_eos:
            scores = scores + 0
            scores[self.speech_token_size] = -float("inf")
        return int(self.sampling(scores, decoded_tokens, sampling))

    def _style_embedding(
        self,
        word_embedding: mx.array,
        duration: int,
        boundary: int,
        tone: int,
        pitch: int,
        energy: int,
    ) -> mx.array:
        values = (
            word_embedding,
            self.duration_embedding(mx.array([duration], dtype=mx.int32)).reshape(-1),
            self.boundary_embedding(mx.array([boundary], dtype=mx.int32)).reshape(-1),
            self.tone_embedding(mx.array([tone], dtype=mx.int32)).reshape(-1),
            self.f0_embedding(mx.array([pitch], dtype=mx.int32)).reshape(-1),
            self.energy_embedding(mx.array([energy], dtype=mx.int32)).reshape(-1),
        )
        return mx.mean(mx.stack(values, axis=0), axis=0).reshape(1, 1, -1)

    def _prepare_prefix(
        self,
        *,
        key: str,
        text: mx.array,
        prompt_speech_token: mx.array,
        word_tokens: Sequence[mx.array],
        starts: Sequence[int],
        durations: Sequence[int],
        boundaries: Sequence[int],
        tones: Sequence[int],
        pitches: Sequence[int],
        energies: Sequence[int],
    ) -> PreparedWordVoicePrefix:
        started = time.perf_counter()
        text_embedding = self.llm.embed_tokens(text)
        word_embeddings = tuple(
            self.llm.embed_tokens(word)[0, 0, :] for word in word_tokens
        )
        style_embeddings = tuple(
            self._style_embedding(
                word_embeddings[index],
                min(max(int(durations[index]), 0), self.max_duration - 1),
                int(boundaries[index]),
                int(tones[index]),
                int(pitches[index]),
                int(energies[index]),
            )
            for index in range(len(word_tokens))
        )

        mutable_durations = list(int(value) for value in durations)
        word_index = 0
        preceding_duration = 0
        prompt_embeddings: list[mx.array] = []
        prompt_length = int(prompt_speech_token.shape[1])
        speech_embeddings = self.speech_embedding(prompt_speech_token)[0]
        for token_index in range(prompt_length):
            if word_index < len(starts) and int(starts[word_index]) == token_index:
                prompt_embeddings.append(self.speech_embedding.weight[self.bound_token])
                prompt_embeddings.append(style_embeddings[word_index].reshape(-1))
                if word_index > 0:
                    mutable_durations[word_index - 1] = preceding_duration
                    preceding_duration = 0
                word_index += 1
            if word_index > 0:
                preceding_duration += 1
            prompt_embeddings.append(speech_embeddings[token_index])
        while word_index < len(starts) and int(starts[word_index]) == prompt_length:
            prompt_embeddings.append(self.speech_embedding.weight[self.bound_token])
            prompt_embeddings.append(style_embeddings[word_index].reshape(-1))
            mutable_durations[word_index - 1] = preceding_duration
            preceding_duration = 0
            word_index += 1
        if prompt_embeddings:
            prompt_embedding = mx.stack(prompt_embeddings, axis=0)[None, :, :]
        else:
            prompt_embedding = mx.zeros((1, 0, self.llm_input_size), dtype=text_embedding.dtype)
        model_input = mx.concatenate(
            [
                self.speech_embedding.weight[self.sos].reshape(1, 1, -1),
                text_embedding,
                self.speech_embedding.weight[self.task_id].reshape(1, 1, -1),
                prompt_embedding,
            ],
            axis=1,
        )
        hidden, cache = self.llm.forward_one_step(model_input, cache=None)
        cache_states = tuple(layer.state for layer in cache)
        arrays = [hidden]
        retained_bytes = int(hidden.nbytes)
        for keys, values in cache_states:
            arrays.extend((keys, values))
            retained_bytes += int(keys.nbytes + values.nbytes)
        arrays.extend(word_embeddings)
        retained_bytes += sum(int(value.nbytes) for value in word_embeddings)
        mx.eval(*arrays)
        if word_index == 0:
            raise ValueError("prepared WordVoice prompt did not match any aligned word start")
        return PreparedWordVoicePrefix(
            owner=self,
            key=key,
            hidden=hidden[:, -1:, :],
            cache_states=cache_states,
            word_embeddings=word_embeddings,
            word_index=word_index,
            preceding_duration=preceding_duration,
            durations=tuple(mutable_durations),
            boundaries=tuple(int(value) for value in boundaries),
            tones=tuple(int(value) for value in tones),
            pitches=tuple(int(value) for value in pitches),
            energies=tuple(int(value) for value in energies),
            final_duration=mutable_durations[word_index - 1],
            final_boundary=int(boundaries[word_index - 1]),
            prepare_seconds=time.perf_counter() - started,
            retained_bytes=retained_bytes,
        )

    @staticmethod
    def _restore_cache(prefix: PreparedWordVoicePrefix) -> list[KVCache]:
        cache = [KVCache() for _ in prefix.cache_states]
        for layer, state in zip(cache, prefix.cache_states):
            layer.state = state
        return cache

    def infer_controlled(
        self,
        *,
        key: str,
        text: mx.array,
        prompt_text: mx.array,
        prompt_speech_token: mx.array,
        word_tokens: Sequence[mx.array],
        starts: Sequence[int],
        durations: Sequence[int],
        boundaries: Sequence[int],
        tones: Sequence[int],
        pitches: Sequence[int],
        energies: Sequence[int],
        sampling: int = 25,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
        better_infer: bool = True,
        use_prepared_prefix: bool = True,
    ) -> ControlledTokenResult:
        """Generate controlled speech tokens with exact one-entry prefix reuse."""
        if not key:
            raise ValueError("a non-empty prepared request fingerprint is required")
        full_text = mx.concatenate([prompt_text, text], axis=1)
        cache_hit = bool(
            use_prepared_prefix
            and self._prepared_prefix is not None
            and self._prepared_prefix.owner is self
            and self._prepared_prefix.key == key
        )
        if cache_hit:
            prefix = self._prepared_prefix
            self._prepared_prefix_hits += 1
            prepare_seconds = 0.0
        else:
            prefix = self._prepare_prefix(
                key=key,
                text=full_text,
                prompt_speech_token=prompt_speech_token,
                word_tokens=word_tokens,
                starts=starts,
                durations=durations,
                boundaries=boundaries,
                tones=tones,
                pitches=pitches,
                energies=energies,
            )
            prepare_seconds = prefix.prepare_seconds
            if use_prepared_prefix:
                self._prepared_prefix = prefix
                self._prepared_prefix_misses += 1

        restored = time.perf_counter()
        cache = self._restore_cache(prefix)
        hidden = prefix.hidden
        mutable_durations = list(prefix.durations)
        mutable_boundaries = list(prefix.boundaries)
        mutable_tones = list(prefix.tones)
        mutable_pitches = list(prefix.pitches)
        mutable_energies = list(prefix.energies)
        word_index = prefix.word_index
        true_duration = prefix.preceding_duration
        final_duration = prefix.final_duration
        final_boundary = prefix.final_boundary
        pauses = [0] * len(word_tokens)
        restore_seconds = time.perf_counter() - restored

        generated: list[int] = []
        target_text_length = int(text.shape[1])
        min_length = int(target_text_length * min_token_text_ratio)
        max_length = int(target_text_length * max_token_text_ratio)
        boundary_embedding = self.speech_embedding.weight[self.bound_token].reshape(1, 1, -1)
        current_input: mx.array | None = None
        for step in range(max_length):
            if step == 0:
                model_output = hidden
            else:
                if current_input is None:
                    raise RuntimeError("WordVoice decoder input was not advanced")
                model_output, cache = self.llm.forward_one_step(current_input, cache=cache)
            logits = self.llm_decoder(model_output[:, -1, :])
            if word_index < len(word_tokens):
                logits = logits + 0
                logits[:, self.eos_token] = -float("inf")
            if better_infer:
                logits = logits + 0
                if true_duration < final_duration:
                    logits[:, list(self.silent_tokens)] = -float("inf")
                    logits[:, self.bound_token] = -float("inf")
                elif true_duration == final_duration:
                    logits[:, self.bound_token] = -float("inf")
                else:
                    allowed = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
                    pause_length = true_duration - final_duration
                    allows_silence = pause_length < self.max_pause_lens[final_boundary]
                    allows_terminal = pause_length > self.max_pause_lens[final_boundary - 1]
                    if allows_silence:
                        allowed[list(self.silent_tokens)] = True
                    if allows_terminal:
                        allowed[self.bound_token] = True
                        allowed[self.eos_token] = True
                    if not allows_silence and not allows_terminal:
                        allowed[self.bound_token] = True
                    logits = mx.where(allowed[None, :], logits, -float("inf"))
            token_id = self._sample(
                logits.reshape(-1),
                generated,
                sampling,
                ignore_eos=step < min_length,
            )
            if token_id in self.silent_tokens and word_index > 0:
                pauses[word_index - 1] += 1
            if better_infer and true_duration == final_duration and final_boundary == 0:
                token_id = self.bound_token
            if word_index == len(word_tokens) and token_id in (
                self.eos_token,
                self.bound_token,
            ):
                break
            if token_id == self.bound_token:
                inner_output, cache = self.llm.forward_one_step(boundary_embedding, cache=cache)
                requested = (
                    mutable_durations[word_index],
                    mutable_boundaries[word_index],
                    mutable_tones[word_index],
                    mutable_pitches[word_index],
                    mutable_energies[word_index],
                )
                masked = (
                    self.max_duration,
                    self.max_boundary,
                    self.max_tone,
                    self.max_f0,
                    self.max_energy,
                )
                if any(value == mask for value, mask in zip(requested, masked)):
                    state = inner_output[:, -1, :]
                    predicted = (
                        int(mx.argmax(self.duration_predictor(state), axis=-1).item()),
                        int(mx.argmax(self.boundary_predictor(state), axis=-1).item()),
                        int(mx.argmax(self.tone_predictor(state), axis=-1).item()),
                        int(mx.argmax(self.f0_predictor(state), axis=-1).item()),
                        int(mx.argmax(self.energy_predictor(state), axis=-1).item()),
                    )
                else:
                    predicted = requested
                final_duration = predicted[0] if requested[0] == self.max_duration else requested[0]
                final_boundary = predicted[1] if requested[1] == self.max_boundary else requested[1]
                final_tone = predicted[2] if requested[2] == self.max_tone else requested[2]
                final_pitch = predicted[3] if requested[3] == self.max_f0 else requested[3]
                final_energy = predicted[4] if requested[4] == self.max_energy else requested[4]
                if better_infer:
                    if requested[1] == self.max_boundary:
                        final_boundary = min(final_boundary, 3)
                    if requested[4] == self.max_energy:
                        final_energy = max(final_energy, min(mutable_energies))
                final_duration = min(max(final_duration, 0), self.max_duration - 1)
                mutable_boundaries[word_index] = final_boundary
                mutable_tones[word_index] = final_tone
                mutable_pitches[word_index] = final_pitch
                mutable_energies[word_index] = final_energy
                current_input = self._style_embedding(
                    prefix.word_embeddings[word_index],
                    final_duration,
                    final_boundary,
                    final_tone,
                    final_pitch,
                    final_energy,
                )
                mutable_durations[word_index - 1] = true_duration
                true_duration = 0
                word_index += 1
            else:
                true_duration += 1
                generated.append(token_id)
                current_input = self.speech_embedding.weight[token_id].reshape(1, 1, -1)
        mutable_durations[-1] = true_duration
        mx.eval(*cache, current_input if current_input is not None else hidden)
        return ControlledTokenResult(
            speech_tokens=tuple(generated),
            durations=tuple(mutable_durations),
            boundaries=tuple(mutable_boundaries),
            tones=tuple(mutable_tones),
            pitches=tuple(mutable_pitches),
            energies=tuple(mutable_energies),
            pauses=tuple(pauses),
            metrics={
                "prepared_prefix_cache": "hit"
                if cache_hit
                else ("miss" if use_prepared_prefix else "disabled"),
                "prepared_prefix_hits": self._prepared_prefix_hits,
                "prepared_prefix_misses": self._prepared_prefix_misses,
                "prepared_prefix_prepare_seconds": round(prepare_seconds, 6),
                "prepared_prefix_restore_seconds": round(restore_seconds, 6),
                "prepared_prefix_retained_bytes": prefix.retained_bytes
                if use_prepared_prefix
                else 0,
            },
        )
