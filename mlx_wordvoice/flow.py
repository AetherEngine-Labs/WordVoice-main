"""MLX flow modulation for WordVoice's word-level acoustic controls."""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.cosyvoice3.flow import CausalMaskedDiffWithDiT, make_pad_mask


class DynamicSinusoidalPosition(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def __call__(self, value: mx.array) -> mx.array:
        length = value.shape[1]
        position = mx.arange(length, dtype=mx.float32)[:, None]
        frequency = mx.exp(
            mx.arange(0, self.dimensions, 2, dtype=mx.float32)
            * (-mx.log(mx.array(10000.0)) / self.dimensions)
        )
        phase = position * frequency[None, :]
        encoding = mx.stack([mx.sin(phase), mx.cos(phase)], axis=-1).reshape(
            length, self.dimensions
        )
        return value + encoding[None, :, :].astype(value.dtype)


class WordVoiceFlow(CausalMaskedDiffWithDiT):
    """CosyVoice3 DiT flow with WordVoice control scale/shift modulation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_bnd = 5
        self.max_tone = 7
        self.max_f0 = 20
        self.max_energy = 20
        control_dimensions = self.input_size // 4
        self.bnd_embed = nn.Embedding(self.max_bnd + 1, control_dimensions)
        self.tone_embed = nn.Embedding(self.max_tone + 1, control_dimensions)
        self.f0_embed = nn.Embedding(self.max_f0 + 1, control_dimensions)
        self.energy_embed = nn.Embedding(self.max_energy + 1, control_dimensions)
        self.word_pos_embed = DynamicSinusoidalPosition(self.input_size)
        self.control_modulator = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.SiLU(),
            nn.Linear(self.input_size, self.input_size * 2),
        )
        self.compilation_enabled = False
        self._compiled_prelookahead = None
        self._compiled_control_modulator = None
        self._compiled_flow_step = None

    def enable_compilation(self) -> None:
        """Compile immutable repeated inference kernels after weights are loaded."""
        if self.compilation_enabled:
            return
        self._compiled_prelookahead = mx.compile(self.pre_lookahead_layer)
        self._compiled_control_modulator = mx.compile(self.control_modulator)

        def flow_step(x, mask, mu, speaker, condition, timestep, delta):
            batch = mu.shape[0]
            x_batched = mx.concatenate([x, x], axis=0)
            mask_batched = mx.concatenate([mask, mask], axis=0)
            mu_batched = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            speaker_batched = mx.concatenate(
                [speaker, mx.zeros_like(speaker)], axis=0
            )
            condition_batched = mx.concatenate(
                [condition, mx.zeros_like(condition)], axis=0
            )
            timestep_batched = mx.broadcast_to(timestep, (2 * batch,))
            derivative = self.decoder.estimator(
                x=x_batched,
                mask=mask_batched,
                mu=mu_batched,
                t=timestep_batched,
                spks=speaker_batched,
                cond=condition_batched,
                streaming=False,
            )
            conditional = derivative[:batch]
            unconditional = derivative[batch:]
            guided = (
                (1.0 + self.decoder.inference_cfg_rate) * conditional
                - self.decoder.inference_cfg_rate * unconditional
            )
            return x + delta * guided

        self._compiled_flow_step = mx.compile(flow_step)
        self.compilation_enabled = True

    def _decode_compiled(
        self,
        *,
        mu: mx.array,
        mask: mx.array,
        speaker: mx.array,
        condition: mx.array,
    ) -> mx.array:
        if self._compiled_flow_step is None:
            raise RuntimeError("WordVoice flow compilation is not enabled")
        if self.decoder._rand_noise is not None:
            value = self.decoder._rand_noise[:, :, : mu.shape[2]].astype(mu.dtype)
        else:
            mx.random.seed(0)
            value = mx.random.normal(shape=mu.shape)
        timesteps = mx.linspace(0, 1, self.n_timesteps + 1)
        if self.decoder.t_scheduler == "cosine":
            timesteps = 1 - mx.cos(timesteps * 0.5 * math.pi)
        flat_mask = mask.squeeze(1)
        for step in range(1, self.n_timesteps + 1):
            value = self._compiled_flow_step(
                value,
                flat_mask,
                mu,
                speaker,
                condition,
                timesteps[step - 1],
                timesteps[step] - timesteps[step - 1],
            )
            mx.eval(value)
        return value.astype(mx.float32)

    @staticmethod
    def _expand_words(features: mx.array, durations: Sequence[int]) -> mx.array:
        expanded = []
        for index, duration in enumerate(durations):
            count = int(duration)
            if count > 0:
                expanded.append(
                    mx.broadcast_to(
                        features[:, index : index + 1, :],
                        (features.shape[0], count, features.shape[-1]),
                    )
                )
        if not expanded:
            raise ValueError("WordVoice flow controls expand to zero speech tokens")
        return mx.concatenate(expanded, axis=1)

    def inference_wordvoice(
        self,
        *,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        start_id: int,
        durations: Sequence[int],
        boundaries: Sequence[int],
        tones: Sequence[int],
        pitches: Sequence[int],
        energies: Sequence[int],
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        streaming: bool = False,
        finalize: bool = True,
    ) -> mx.array:
        if token.shape[0] != 1:
            raise ValueError("WordVoice MLX inference requires batch size one")
        total_expected = int(prompt_token_len.item() + token_len.item())
        control_ids = (
            [self.max_bnd, *map(int, boundaries)],
            [self.max_tone, *map(int, tones)],
            [self.max_f0, *map(int, pitches)],
            [self.max_energy, *map(int, energies)],
        )
        boundary_embedding = self.bnd_embed(mx.array([control_ids[0]], dtype=mx.int32))
        tone_embedding = self.tone_embed(mx.array([control_ids[1]], dtype=mx.int32))
        pitch_embedding = self.f0_embed(mx.array([control_ids[2]], dtype=mx.int32))
        energy_embedding = self.energy_embed(mx.array([control_ids[3]], dtype=mx.int32))
        word_features = self.word_pos_embed(
            mx.concatenate(
                [boundary_embedding, tone_embedding, pitch_embedding, energy_embedding],
                axis=-1,
            )
        )
        expanded_controls = self._expand_words(
            word_features, [int(start_id), *map(int, durations)]
        )
        if expanded_controls.shape[1] > total_expected:
            expanded_controls = expanded_controls[:, :total_expected, :]
        if expanded_controls.shape[1] != total_expected:
            raise ValueError(
                "WordVoice flow control duration mismatch: "
                f"expanded {expanded_controls.shape[1]}, expected {total_expected}"
            )

        embedding = embedding / mx.sqrt(
            mx.sum(embedding * embedding, axis=-1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)
        complete_token = mx.concatenate([prompt_token, token], axis=1)
        complete_length = prompt_token_len + token_len
        mask = ~make_pad_mask(complete_length, complete_token.shape[1])
        mask = mask[:, :, None].astype(embedding.dtype)
        token_embedding = self.input_embedding(
            mx.clip(complete_token, 0, self.vocab_size - 1)
        ) * mask
        if finalize:
            if self._compiled_prelookahead is not None:
                hidden = self._compiled_prelookahead(token_embedding)
            else:
                hidden = self.pre_lookahead_layer(token_embedding)
            sliced_controls = expanded_controls
        else:
            hidden = self.pre_lookahead_layer(
                token_embedding[:, : -self.pre_lookahead_len, :],
                context=token_embedding[:, -self.pre_lookahead_len :, :],
            )
            sliced_controls = expanded_controls[:, : -self.pre_lookahead_len, :]
        hidden = mx.repeat(hidden, self.token_mel_ratio, axis=1)
        sliced_controls = mx.repeat(sliced_controls, self.token_mel_ratio, axis=1)
        if self._compiled_control_modulator is not None:
            scale_shift = self._compiled_control_modulator(sliced_controls)
        else:
            scale_shift = self.control_modulator(sliced_controls)
        scale, shift = mx.split(scale_shift, 2, axis=-1)
        hidden = hidden * (1.0 + mx.tanh(scale)) + shift

        prompt_frames = prompt_feat.shape[1]
        generated_frames = hidden.shape[1] - prompt_frames
        if generated_frames <= 0:
            raise ValueError(
                "WordVoice flow produced no target frames after prompt conditioning"
            )
        condition = mx.concatenate(
            [
                prompt_feat,
                mx.zeros((1, generated_frames, self.output_size), dtype=hidden.dtype),
            ],
            axis=1,
        )
        decoder_mu = mx.swapaxes(hidden, 1, 2)
        decoder_mask = mx.ones((1, 1, hidden.shape[1]), dtype=mx.float32)
        decoder_condition = mx.swapaxes(condition, 1, 2)
        if self.compilation_enabled:
            if streaming:
                raise ValueError("compiled WordVoice flow currently supports non-stream inference")
            feature = self._decode_compiled(
                mu=decoder_mu,
                mask=decoder_mask,
                speaker=embedding,
                condition=decoder_condition,
            )
        else:
            feature, _ = self.decoder(
                mu=decoder_mu,
                mask=decoder_mask,
                spks=embedding,
                cond=decoder_condition,
                n_timesteps=self.n_timesteps,
                streaming=streaming,
            )
        feature = feature[:, :, prompt_frames:]
        if feature.shape[2] != generated_frames:
            raise RuntimeError(
                "WordVoice flow output frame mismatch: "
                f"actual {feature.shape[2]}, expected {generated_frames}"
            )
        mx.eval(feature)
        return feature.astype(mx.float32)
