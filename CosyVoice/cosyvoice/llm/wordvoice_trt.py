"""Fail-closed TensorRT decode-step adapter for WordVoice's Qwen2 decoder."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


MANIFEST_SCHEMA = "wordvoice.tensorrt.decoder.v1"
ENGINE_FILENAME = "decoder.plan"
MANIFEST_FILENAME = "manifest.json"
LAYERED_CACHE_LAYOUT = "layered"
FLAT_CACHE_LAYOUT = "token-major-flat-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_tensor_name(kind: str, layer: int, *, present: bool = False) -> str:
    prefix = "present" if present else "past"
    return f"{prefix}_{kind}_{layer}"


def _flat_cache_view(
    buffer: torch.Tensor,
    *,
    offset: int = 0,
    length: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Expose an exact-shaped contiguous view into a flat cache buffer.

    TensorRT receives only the device pointer, while the returned view keeps
    the dynamic shape expected by the rest of the WordVoice cache contract.
    A sliced view of a full-capacity four-dimensional tensor would be
    non-contiguous, but a range within a flat allocation can be reshaped
    without introducing a stride change.
    """

    if buffer.ndim != 1:
        raise RuntimeError(
            "WordVoice TensorRT gate failed; stage=cache-buffer; "
            f"expected=one-dimensional-buffer; actual_shape={tuple(buffer.shape)}"
        )
    element_count = num_kv_heads * length * head_dim
    if offset < 0 or offset + element_count > buffer.numel():
        raise RuntimeError(
            "WordVoice TensorRT gate failed; stage=cache-buffer-capacity; "
            f"expected_end={offset + element_count}; actual={buffer.numel()}"
        )
    view = buffer.narrow(0, offset, element_count).view(
        1, num_kv_heads, length, head_dim
    )
    if not view.is_contiguous():
        raise RuntimeError(
            "WordVoice TensorRT gate failed; stage=cache-buffer-layout; "
            f"expected=contiguous; actual_strides={view.stride()}"
        )
    return view


def _flat_token_major_view(
    buffer: torch.Tensor,
    *,
    length: int,
    channels: int,
    head_dim: int,
) -> torch.Tensor:
    """Expose a contiguous ``[batch, tokens, channels, head_dim]`` prefix."""

    element_count = length * channels * head_dim
    if buffer.ndim != 1 or element_count > buffer.numel():
        raise RuntimeError(
            "WordVoice TensorRT gate failed; stage=flat-cache-capacity; "
            f"expected_end={element_count}; actual={buffer.numel()}"
        )
    view = buffer.narrow(0, 0, element_count).view(
        1, length, channels, head_dim
    )
    if not view.is_contiguous():
        raise RuntimeError(
            "WordVoice TensorRT gate failed; stage=flat-cache-layout; "
            f"expected=contiguous; actual_strides={view.stride()}"
        )
    return view


def _flat_layer_cache_view(
    buffer: torch.Tensor,
    *,
    length: int,
    layer: int,
    kind_index: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Expose one legacy-shaped cache view from a token-major flat cache."""

    channels = num_layers * 2 * num_kv_heads
    token_major = _flat_token_major_view(
        buffer,
        length=length,
        channels=channels,
        head_dim=head_dim,
    )
    channel_offset = (layer * 2 + kind_index) * num_kv_heads
    return token_major[:, :, channel_offset : channel_offset + num_kv_heads, :].permute(
        0, 2, 1, 3
    )


@dataclass
class _ExecutionContextState:
    context: Any
    fixed_input_shapes_set: bool = False
    last_cache_shape: tuple[int, ...] | None = None
    validated_output_shapes: bool = False
    bound_input_cache_key: tuple[str, int] | None = None
    output_addresses_bound: bool = False


@dataclass(frozen=True)
class _FlatCacheState:
    """Internal cache handle that avoids rebuilding 48 legacy views per token."""

    buffer: torch.Tensor
    slot: int
    length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                _flat_layer_cache_view(
                    self.buffer,
                    length=self.length,
                    layer=layer,
                    kind_index=0,
                    num_layers=self.num_layers,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                ),
                _flat_layer_cache_view(
                    self.buffer,
                    length=self.length,
                    layer=layer,
                    kind_index=1,
                    num_layers=self.num_layers,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                ),
            )
            for layer in range(self.num_layers)
        )

    def __iter__(self):
        return iter(self.to_legacy_cache())


class WordVoiceTensorRTDecoder:
    """Run only cached single-token decoder steps in a TensorRT engine.

    PyTorch remains responsible for the one prefill call, embeddings, sampling,
    token masking, and WordVoice's five prosody-control heads. This adapter owns
    the repeated Qwen2 transformer step after a non-empty KV cache exists.
    """

    def __init__(self, runtime_dir: str | Path, checkpoint_path: str | Path):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=runtime-load; "
                "expected=CUDA available; actual=unavailable"
            )
        self.runtime_dir = Path(runtime_dir).resolve()
        self.engine_path = self.runtime_dir / ENGINE_FILENAME
        self.manifest_path = self.runtime_dir / MANIFEST_FILENAME
        checkpoint = Path(checkpoint_path).resolve()
        for required in (self.engine_path, self.manifest_path, checkpoint):
            if not required.is_file():
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=runtime-load; "
                    f"expected=regular-file; actual={required}"
                )

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=manifest-schema; "
                f"expected={MANIFEST_SCHEMA}; actual={manifest.get('schema')!r}; "
                f"manifest={self.manifest_path}"
            )
        self.manifest = manifest
        self.num_layers = self._positive_int("num_hidden_layers")
        self.num_kv_heads = self._positive_int("num_key_value_heads")
        self.head_dim = self._positive_int("head_dim")
        self.hidden_size = self._positive_int("hidden_size")
        self.max_past_tokens = self._positive_int("max_past_tokens")
        self.precision = manifest.get("precision")
        self.cache_layout = manifest.get("cache_layout", LAYERED_CACHE_LAYOUT)
        if self.cache_layout not in {LAYERED_CACHE_LAYOUT, FLAT_CACHE_LAYOUT}:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-layout-mode; "
                f"expected={LAYERED_CACHE_LAYOUT}|{FLAT_CACHE_LAYOUT}; "
                f"actual={self.cache_layout!r}"
            )
        if self.precision not in {"fp32", "autocast_fp16"}:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=precision; "
                f"expected=fp32|autocast_fp16; actual={self.precision!r}"
            )
        self.dtype = (
            torch.float16 if self.precision == "autocast_fp16" else torch.float32
        )

        expected_engine_sha = manifest.get("engine_sha256")
        actual_engine_sha = sha256(self.engine_path)
        if actual_engine_sha != expected_engine_sha:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=engine-hash; "
                f"expected={expected_engine_sha}; actual={actual_engine_sha}; "
                f"engine={self.engine_path}"
            )
        expected_checkpoint_sha = manifest.get("wordvoice_llm_sha256")
        actual_checkpoint_sha = sha256(checkpoint)
        if actual_checkpoint_sha != expected_checkpoint_sha:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=checkpoint-hash; "
                f"expected={expected_checkpoint_sha}; actual={actual_checkpoint_sha}; "
                f"checkpoint={checkpoint}"
            )

        import tensorrt as trt

        if trt.__version__ != manifest.get("tensorrt_version"):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=tensorrt-version; "
                f"expected={manifest.get('tensorrt_version')!r}; "
                f"actual={trt.__version__!r}"
            )
        capability = torch.cuda.get_device_capability()
        expected_capability = manifest.get("cuda_compute_capability")
        actual_capability = f"{capability[0]}.{capability[1]}"
        if actual_capability != expected_capability:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=gpu-capability; "
                f"expected={expected_capability!r}; actual={actual_capability!r}"
            )

        logger = trt.Logger(trt.Logger.WARNING)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=engine-deserialize; "
                f"engine={self.engine_path}"
            )
        context_states: list[_ExecutionContextState] = []
        for context_index in range(2):
            context = self.engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=context-create; "
                    f"context_index={context_index}; engine={self.engine_path}"
                )
            context_states.append(_ExecutionContextState(context))
        self._context_states = context_states
        # Keep the first context available for diagnostics and compatibility;
        # decode steps select the direction-specific state below.
        self.context = context_states[0].context
        expected_names = self._expected_tensor_names()
        actual_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        if actual_names != expected_names:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=io-contract; "
                f"expected={sorted(expected_names)}; actual={sorted(actual_names)}"
            )

        # Reuse timing events across runs, but do not synchronize the host on
        # every decode token.  The caller already consumes the result on the
        # same CUDA stream; per-token synchronization only serializes the
        # Python hot loop with the GPU.
        self._timing_event_pairs: list[tuple[Any, Any]] = []
        self._timing_event_cursor = 0
        self._device_seconds = 0.0
        self._execution_lock = threading.Lock()
        self._host_seconds = 0.0
        self._steps = 0
        self._validated_mask_address: int | None = None
        self._parity: dict[str, float] | None = None
        self._cache_buffer_slots: list[torch.Tensor | None] = [None, None]
        self._next_cache_slot = 0
        self._cache_buffer_allocations = 0
        self._flat_cache_buffer_slots: list[torch.Tensor | None] = [None, None]
        self._flat_eager_cache_buffer: torch.Tensor | None = None
        self._flat_cache_buffer_allocations = 0
        self._hidden_buffer_slots: list[torch.Tensor | None] = [None, None]
        self._hidden_buffer_allocations = 0
        self._input_buffer: torch.Tensor | None = None
        self._input_buffer_allocations = 0

    def _positive_int(self, name: str) -> int:
        value = self.manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=manifest-value; "
                f"field={name}; expected=positive-integer; actual={value!r}"
            )
        return value

    def _execute_async(
        self,
        context: Any,
        inputs: torch.Tensor,
        *,
        started: float,
        past_tokens: int,
    ) -> None:
        current_stream = torch.cuda.current_stream(inputs.device)
        if self._timing_event_cursor == len(self._timing_event_pairs):
            self._timing_event_pairs.append(
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            )
        timing_start, timing_end = self._timing_event_pairs[self._timing_event_cursor]
        timing_start.record(current_stream)
        if not context.execute_async_v3(current_stream.cuda_stream):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=execute; "
                f"engine={self.engine_path}; past_tokens={past_tokens}"
            )
        timing_end.record(current_stream)
        self._timing_event_cursor += 1
        self._host_seconds += time.perf_counter() - started
        self._steps += 1

    def _flush_device_timing(self) -> None:
        if self._timing_event_cursor == 0:
            return
        # Synchronize once per generation/metrics window, after all dependent
        # decode work has already been enqueued on the same stream.
        self._timing_event_pairs[self._timing_event_cursor - 1][1].synchronize()
        self._device_seconds += sum(
            start.elapsed_time(end) / 1000.0
            for start, end in self._timing_event_pairs[: self._timing_event_cursor]
        )
        self._timing_event_cursor = 0

    def _expected_tensor_names(self) -> set[str]:
        if self.cache_layout == FLAT_CACHE_LAYOUT:
            return {"inputs_embeds", "past_cache", "hidden_state", "present_cache"}
        names = {"inputs_embeds", "hidden_state"}
        for layer in range(self.num_layers):
            for kind in ("key", "value"):
                names.add(cache_tensor_name(kind, layer))
                names.add(cache_tensor_name(kind, layer, present=True))
        return names

    def _cache_slot_for_legacy_cache(self, legacy_cache: Any) -> int | None:
        if (
            not legacy_cache
            or not isinstance(legacy_cache[0], (tuple, list))
            or not legacy_cache[0]
        ):
            return None
        first_tensor = legacy_cache[0][0]
        if not isinstance(first_tensor, torch.Tensor):
            return None
        first_ptr = first_tensor.data_ptr()
        for index, buffers in enumerate(self._cache_buffer_slots):
            if buffers is not None and buffers.data_ptr() == first_ptr:
                return index
        return None

    def _ensure_cache_buffer_slot(
        self, slot: int, *, device: torch.device
    ) -> torch.Tensor:
        buffers = self._cache_buffer_slots[slot]
        if buffers is None:
            cache_tensor_elements = (
                self.num_kv_heads
                * (self.max_past_tokens + 1)
                * self.head_dim
            )
            buffers = torch.empty(
                (cache_tensor_elements * self.num_layers * 2,),
                device=device,
                dtype=self.dtype,
            )
            self._cache_buffer_slots[slot] = buffers
            self._cache_buffer_allocations += 1
        elif buffers.device != device or buffers.dtype != self.dtype:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-buffer-device; "
                f"expected={device}/{self.dtype}; "
                f"actual={buffers.device}/{buffers.dtype}"
            )
        return buffers

    def _flat_cache_channels(self) -> int:
        return self.num_layers * 2 * self.num_kv_heads

    def _flat_cache_slot_for_legacy_cache(self, legacy_cache: Any) -> int | None:
        if (
            not legacy_cache
            or not isinstance(legacy_cache[0], (tuple, list))
            or not legacy_cache[0]
        ):
            return None
        first_tensor = legacy_cache[0][0]
        if not isinstance(first_tensor, torch.Tensor):
            return None
        first_ptr = first_tensor.data_ptr()
        for index, buffer in enumerate(self._flat_cache_buffer_slots):
            if buffer is not None and buffer.data_ptr() == first_ptr:
                return index
        return None

    def _ensure_flat_cache_buffer_slot(
        self, slot: int, *, device: torch.device
    ) -> torch.Tensor:
        buffer = self._flat_cache_buffer_slots[slot]
        if buffer is None:
            element_count = (
                (self.max_past_tokens + 1)
                * self._flat_cache_channels()
                * self.head_dim
            )
            buffer = torch.empty(
                (element_count,), device=device, dtype=self.dtype
            )
            self._flat_cache_buffer_slots[slot] = buffer
            self._flat_cache_buffer_allocations += 1
        elif buffer.device != device or buffer.dtype != self.dtype:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=flat-cache-device; "
                f"expected={device}/{self.dtype}; "
                f"actual={buffer.device}/{buffer.dtype}"
            )
        return buffer

    def _ensure_flat_eager_cache_buffer(
        self, *, device: torch.device
    ) -> torch.Tensor:
        if self._flat_eager_cache_buffer is None:
            element_count = (
                (self.max_past_tokens + 1)
                * self._flat_cache_channels()
                * self.head_dim
            )
            self._flat_eager_cache_buffer = torch.empty(
                (element_count,), device=device, dtype=self.dtype
            )
            self._flat_cache_buffer_allocations += 1
        elif (
            self._flat_eager_cache_buffer.device != device
            or self._flat_eager_cache_buffer.dtype != self.dtype
        ):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=flat-eager-cache-device; "
                f"expected={device}/{self.dtype}; "
                f"actual={self._flat_eager_cache_buffer.device}/"
                f"{self._flat_eager_cache_buffer.dtype}"
            )
        return self._flat_eager_cache_buffer

    def _ensure_hidden_buffer_slot(
        self, slot: int, *, device: torch.device
    ) -> torch.Tensor:
        hidden = self._hidden_buffer_slots[slot]
        if hidden is None:
            hidden = torch.empty(
                (1, 1, self.hidden_size), device=device, dtype=self.dtype
            )
            self._hidden_buffer_slots[slot] = hidden
            self._hidden_buffer_allocations += 1
        elif hidden.device != device or hidden.dtype != self.dtype:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=hidden-buffer-device; "
                f"expected={device}/{self.dtype}; "
                f"actual={hidden.device}/{hidden.dtype}"
            )
        return hidden

    def _prepare_inputs(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.dtype == self.dtype and inputs_embeds.is_contiguous():
            return inputs_embeds
        if self._input_buffer is None:
            self._input_buffer = torch.empty(
                tuple(inputs_embeds.shape),
                device=inputs_embeds.device,
                dtype=self.dtype,
            )
            self._input_buffer_allocations += 1
        elif (
            self._input_buffer.device != inputs_embeds.device
            or self._input_buffer.dtype != self.dtype
            or tuple(self._input_buffer.shape) != tuple(inputs_embeds.shape)
        ):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=input-buffer-device; "
                f"expected={inputs_embeds.device}/{self.dtype}/{tuple(inputs_embeds.shape)}; "
                f"actual={self._input_buffer.device}/{self._input_buffer.dtype}/"
                f"{tuple(self._input_buffer.shape)}"
            )
        self._input_buffer.copy_(inputs_embeds)
        return self._input_buffer

    def _pack_eager_cache_to_flat(
        self, legacy_cache: Any, *, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        if not isinstance(legacy_cache, (tuple, list)) or len(legacy_cache) != self.num_layers:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-layout; "
                f"expected={self.num_layers}-layer-cache"
            )
        tensors: list[torch.Tensor] = []
        past_tokens: int | None = None
        for layer, pair in enumerate(legacy_cache):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-layout; "
                    f"layer={layer}; expected=key-value-pair"
                )
            for kind, tensor in zip(("key", "value"), pair):
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    raise RuntimeError(
                        "WordVoice TensorRT gate failed; stage=cache-device; "
                        f"layer={layer}; kind={kind}; expected=CUDA-tensor"
                    )
                shape = tuple(tensor.shape)
                if (
                    len(shape) != 4
                    or shape[0] != 1
                    or shape[1] != self.num_kv_heads
                    or shape[3] != self.head_dim
                ):
                    raise RuntimeError(
                        "WordVoice TensorRT gate failed; stage=cache-shape; "
                        f"layer={layer}; kind={kind}; actual={shape}"
                    )
                if past_tokens is None:
                    past_tokens = shape[2]
                elif shape[2] != past_tokens:
                    raise RuntimeError(
                        "WordVoice TensorRT gate failed; stage=cache-length; "
                        f"expected={past_tokens}; actual={shape[2]}; layer={layer}"
                    )
                if tensor.dtype != self.dtype:
                    tensor = tensor.to(dtype=self.dtype)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                tensors.append(tensor)
        if past_tokens is None or not 1 <= past_tokens <= self.max_past_tokens:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-length; "
                f"expected=1..{self.max_past_tokens}; actual={past_tokens}"
            )
        buffer = self._ensure_flat_eager_cache_buffer(device=device)
        destination = _flat_token_major_view(
            buffer,
            length=past_tokens,
            channels=self._flat_cache_channels(),
            head_dim=self.head_dim,
        )
        for index, tensor in enumerate(tensors):
            channel_offset = index * self.num_kv_heads
            destination[
                :, :, channel_offset : channel_offset + self.num_kv_heads, :
            ].copy_(tensor.permute(0, 2, 1, 3))
        return destination, past_tokens

    def validate_against_eager(self, qwen_model: torch.nn.Module) -> dict[str, float]:
        """Prove the loaded engine matches this checkpoint before activation."""

        dtype = self.dtype
        prefill = torch.linspace(
            -0.125,
            0.125,
            steps=4 * self.hidden_size,
            device="cuda",
            dtype=dtype,
        ).reshape(1, 4, self.hidden_size)
        next_input = torch.linspace(
            0.125,
            -0.125,
            steps=self.hidden_size,
            device="cuda",
            dtype=dtype,
        ).reshape(1, 1, self.hidden_size)
        precision_context = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.precision == "autocast_fp16"
            else torch.cuda.amp.autocast(enabled=False)
        )
        with torch.inference_mode(), precision_context:
            prefill_output = qwen_model(
                inputs_embeds=prefill,
                attention_mask=torch.ones(1, 4, device="cuda", dtype=torch.bool),
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )
            prefill_legacy = prefill_output.past_key_values.to_legacy_cache()
            eager_cache_input = DynamicCache.from_legacy_cache(
                tuple((key.clone(), value.clone()) for key, value in prefill_legacy)
            )
            native_cache_input = DynamicCache.from_legacy_cache(
                tuple((key.clone(), value.clone()) for key, value in prefill_legacy)
            )
            eager_output = qwen_model(
                inputs_embeds=next_input,
                attention_mask=torch.ones(1, 1, device="cuda", dtype=torch.bool),
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                past_key_values=eager_cache_input,
            )
        native_hidden, native_cache = self.forward_one_step(
            next_input,
            torch.ones(1, 1, 1, device="cuda", dtype=torch.bool),
            native_cache_input,
        )
        torch.cuda.synchronize()
        eager_hidden = eager_output.hidden_states[-1][:, -1:, :]
        hidden_error = float(
            (native_hidden.float() - eager_hidden.float()).abs().max().item()
        )
        hidden_rmse = float(
            torch.sqrt(
                torch.mean((native_hidden.float() - eager_hidden.float()).square())
            ).item()
        )
        eager_cache = eager_output.past_key_values
        if hasattr(eager_cache, "to_legacy_cache"):
            eager_cache = eager_cache.to_legacy_cache()
        cache_error = max(
            float((actual.float() - expected.float()).abs().max().item())
            for actual_layer, expected_layer in zip(native_cache, eager_cache)
            for actual, expected in zip(actual_layer, expected_layer)
        )
        cache_rmse = max(
            float(
                torch.sqrt(torch.mean((actual.float() - expected.float()).square())).item()
            )
            for actual_layer, expected_layer in zip(native_cache, eager_cache)
            for actual, expected in zip(actual_layer, expected_layer)
        )
        finite = all(
            torch.isfinite(tensor).all().item()
            for tensor in (native_hidden, eager_hidden)
        ) and all(
            torch.isfinite(tensor).all().item()
            for layer in native_cache
            for tensor in layer
        )
        maximum_tolerance = 0.01 if self.precision == "fp32" else 0.25
        rmse_tolerance = 0.001 if self.precision == "fp32" else 0.025
        if (
            not finite
            or hidden_error > maximum_tolerance
            or cache_error > maximum_tolerance
            or hidden_rmse > rmse_tolerance
            or cache_rmse > rmse_tolerance
        ):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=eager-parity; "
                f"expected=finite and max_abs_error<={maximum_tolerance} "
                f"and rmse<={rmse_tolerance}; "
                f"hidden_max_abs_error={hidden_error:.6f}; "
                f"hidden_rmse={hidden_rmse:.6f}; "
                f"cache_max_abs_error={cache_error:.6f}; "
                f"cache_rmse={cache_rmse:.6f}; engine={self.engine_path}"
            )
        self._parity = {
            "hidden_max_abs_error": round(hidden_error, 6),
            "hidden_rmse": round(hidden_rmse, 6),
            "cache_max_abs_error": round(cache_error, 6),
            "cache_rmse": round(cache_rmse, 6),
        }
        self.reset_metrics()
        return dict(self._parity)

    def forward_one_step(
        self, inputs_embeds: torch.Tensor, masks: torch.Tensor, cache: Any
    ) -> tuple[torch.Tensor, Any]:
        """Run the single shared TensorRT context without overlapping calls."""
        with self._execution_lock:
            return self._forward_one_step(inputs_embeds, masks, cache)

    def _forward_one_step(
        self, inputs_embeds: torch.Tensor, masks: torch.Tensor, cache: Any
    ) -> tuple[torch.Tensor, Any]:
        if self.cache_layout == FLAT_CACHE_LAYOUT:
            return self._forward_flat_cache_one_step(inputs_embeds, masks, cache)
        return self._forward_layered_cache_one_step(inputs_embeds, masks, cache)

    def _forward_layered_cache_one_step(
        self, inputs_embeds: torch.Tensor, masks: torch.Tensor, cache: Any
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        started = time.perf_counter()
        legacy_cache = (
            cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        )
        if not isinstance(legacy_cache, (tuple, list)) or len(legacy_cache) != self.num_layers:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-layout; "
                f"expected={self.num_layers}-layer-cache; "
                f"actual={type(legacy_cache).__name__}"
            )
        if tuple(inputs_embeds.shape) != (1, 1, self.hidden_size):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=input-shape; "
                f"expected={(1, 1, self.hidden_size)}; actual={tuple(inputs_embeds.shape)}"
            )
        if not inputs_embeds.is_cuda:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=input-device; "
                "expected=CUDA tensor; actual=CPU tensor"
            )
        if self._validated_mask_address != masks.data_ptr():
            if (
                tuple(masks.shape) != (1, 1, 1)
                or masks.dtype != torch.bool
                or not bool(masks.item())
            ):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=decode-mask; "
                    "expected=single true causal decode mask; "
                    f"actual_shape={tuple(masks.shape)}; actual_dtype={masks.dtype}"
                )
            self._validated_mask_address = masks.data_ptr()
        inputs = self._prepare_inputs(inputs_embeds)
        cache_inputs: list[torch.Tensor] = []
        past_tokens: int | None = None
        cache_shape: tuple[int, ...] | None = None
        input_cache_slot = self._cache_slot_for_legacy_cache(legacy_cache)
        output_slot = (
            self._next_cache_slot
            if input_cache_slot is None
            else 1 - input_cache_slot
        )
        self._next_cache_slot = 1 - output_slot
        context_state = self._context_states[output_slot]
        context = context_state.context
        input_cache_key: tuple[str, int]
        pooled_cache_fast_path = (
            input_cache_slot is not None
            and context_state.bound_input_cache_key == ("slot", input_cache_slot)
        )
        if pooled_cache_fast_path:
            # This cache was produced by this decoder and its direction-specific
            # addresses were already validated and bound. Only the dynamic
            # sequence length changes between these steps.
            pair = legacy_cache[0]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-layout; "
                    "expected=key-value-pair on pooled fast path"
                )
            first_tensor = pair[0]
            if not isinstance(first_tensor, torch.Tensor) or not first_tensor.is_cuda:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-device; "
                    "expected=CUDA tensor on pooled fast path"
                )
            cache_shape = tuple(first_tensor.shape)
            if (
                len(cache_shape) != 4
                or cache_shape[0] != 1
                or cache_shape[1] != self.num_kv_heads
                or cache_shape[3] != self.head_dim
            ):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-shape; "
                    f"actual={cache_shape} on pooled fast path"
                )
            past_tokens = cache_shape[2]
            if input_cache_slot is None:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-slot; "
                    "expected=registered pooled cache slot"
                )
            input_cache_key = ("slot", input_cache_slot)
        else:
            for layer, pair in enumerate(legacy_cache):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise RuntimeError(
                        "WordVoice TensorRT gate failed; stage=cache-layout; "
                        f"layer={layer}; expected=key-value-pair; actual={type(pair).__name__}"
                    )
                for kind, tensor in zip(("key", "value"), pair):
                    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                        raise RuntimeError(
                            "WordVoice TensorRT gate failed; stage=cache-device; "
                            f"layer={layer}; kind={kind}; expected=CUDA-tensor"
                        )
                    shape = tuple(tensor.shape)
                    if (
                        len(shape) != 4
                        or shape[0] != 1
                        or shape[1] != self.num_kv_heads
                        or shape[3] != self.head_dim
                    ):
                        raise RuntimeError(
                            "WordVoice TensorRT gate failed; stage=cache-shape; "
                            f"layer={layer}; kind={kind}; actual={shape}"
                        )
                    if past_tokens is None:
                        past_tokens = shape[2]
                    elif shape[2] != past_tokens:
                        raise RuntimeError(
                            "WordVoice TensorRT gate failed; stage=cache-length; "
                            f"expected={past_tokens}; actual={shape[2]}; layer={layer}"
                        )
                    if tensor.dtype != self.dtype:
                        tensor = tensor.to(dtype=self.dtype)
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    cache_inputs.append(tensor)
            cache_shape = tuple(cache_inputs[0].shape)
            input_cache_key = (
                ("slot", input_cache_slot)
                if input_cache_slot is not None
                else ("ptr", cache_inputs[0].data_ptr())
            )
        if past_tokens is None or not 1 <= past_tokens <= self.max_past_tokens:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-length; "
                f"expected=1..{self.max_past_tokens}; actual={past_tokens}"
            )

        if not context_state.fixed_input_shapes_set:
            # The token input has a fixed shape for this batch-one decoder.
            # TensorRT retains it for the lifetime of the execution context.
            if not context.set_input_shape("inputs_embeds", tuple(inputs.shape)):
                raise RuntimeError("WordVoice TensorRT failed to set inputs_embeds shape")
            context_state.fixed_input_shapes_set = True
        context.set_tensor_address("inputs_embeds", inputs.data_ptr())
        if cache_shape is None:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-shape; "
                "expected=validated cache shape"
            )
        if context_state.last_cache_shape != cache_shape:
            # Cache length changes only when a new token is appended. Repeated
            # lengths (for example across lines) do not need another shape
            # negotiation; cache addresses are rebound only when their source
            # slot or eager-cache pointer changes.
            index = 0
            for layer in range(self.num_layers):
                for kind in ("key", "value"):
                    name = cache_tensor_name(kind, layer)
                    if not context.set_input_shape(name, cache_shape):
                        raise RuntimeError(
                            "WordVoice TensorRT failed to set cache input shape; "
                            f"tensor={name}; shape={cache_shape}"
                        )
                    index += 1
            context_state.last_cache_shape = cache_shape
        if context_state.bound_input_cache_key != input_cache_key:
            index = 0
            for layer in range(self.num_layers):
                for kind in ("key", "value"):
                    context.set_tensor_address(
                        cache_tensor_name(kind, layer), cache_inputs[index].data_ptr()
                    )
                    index += 1
            context_state.bound_input_cache_key = input_cache_key

        hidden = self._ensure_hidden_buffer_slot(output_slot, device=inputs.device)
        # Present shapes are deterministic from the negotiated cache length.
        # Validate the first execution, then keep the hot path free of 49 host
        # shape queries per token while retaining address and execute checks.
        if not context_state.validated_output_shapes:
            hidden_shape = tuple(context.get_tensor_shape("hidden_state"))
            if hidden_shape != tuple(hidden.shape):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=output-shape; "
                    f"tensor=hidden_state; expected={tuple(hidden.shape)}; actual={hidden_shape}"
                )
        context.set_tensor_address("hidden_state", hidden.data_ptr())
        output_buffers = self._ensure_cache_buffer_slot(
            output_slot, device=inputs.device
        )
        present: list[tuple[torch.Tensor, torch.Tensor]] = []
        output_length = past_tokens + 1
        cache_tensor_capacity = (
            self.num_kv_heads * (self.max_past_tokens + 1) * self.head_dim
        )
        index = 0
        for layer in range(self.num_layers):
            pair: list[torch.Tensor] = []
            for kind in ("key", "value"):
                tensor = _flat_cache_view(
                    output_buffers,
                    offset=index * cache_tensor_capacity,
                    length=output_length,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                )
                name = cache_tensor_name(kind, layer, present=True)
                if not context_state.validated_output_shapes:
                    actual_shape = tuple(context.get_tensor_shape(name))
                    if actual_shape != tuple(tensor.shape):
                        raise RuntimeError(
                            "WordVoice TensorRT gate failed; stage=output-shape; "
                            f"tensor={name}; expected={tuple(tensor.shape)}; "
                            f"actual={actual_shape}"
                        )
                if not context_state.output_addresses_bound:
                    context.set_tensor_address(name, tensor.data_ptr())
                pair.append(tensor)
                index += 1
            present.append((pair[0], pair[1]))
        context_state.output_addresses_bound = True
        context_state.validated_output_shapes = True

        self._execute_async(
            context,
            inputs,
            started=started,
            past_tokens=past_tokens,
        )
        return hidden, tuple(present)

    def _forward_flat_cache_one_step(
        self, inputs_embeds: torch.Tensor, masks: torch.Tensor, cache: Any
    ) -> tuple[torch.Tensor, _FlatCacheState]:
        started = time.perf_counter()
        flat_state = cache if isinstance(cache, _FlatCacheState) else None
        legacy_cache = None
        if flat_state is None:
            legacy_cache = (
                cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
            )
            if (
                not isinstance(legacy_cache, (tuple, list))
                or len(legacy_cache) != self.num_layers
            ):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-layout; "
                    f"expected={self.num_layers}-layer-cache"
                )
        elif (
            flat_state.num_layers != self.num_layers
            or flat_state.num_kv_heads != self.num_kv_heads
            or flat_state.head_dim != self.head_dim
        ):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=flat-cache-state; "
                "expected=matching-engine-metadata; "
                f"actual=layers:{flat_state.num_layers}; "
                f"kv_heads:{flat_state.num_kv_heads}; head_dim:{flat_state.head_dim}"
            )
        if tuple(inputs_embeds.shape) != (1, 1, self.hidden_size):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=input-shape; "
                f"expected={(1, 1, self.hidden_size)}; actual={tuple(inputs_embeds.shape)}"
            )
        if not inputs_embeds.is_cuda:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=input-device; "
                "expected=CUDA tensor; actual=CPU tensor"
            )
        if self._validated_mask_address != masks.data_ptr():
            if (
                tuple(masks.shape) != (1, 1, 1)
                or masks.dtype != torch.bool
                or not bool(masks.item())
            ):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=decode-mask; "
                    "expected=single true causal decode mask; "
                    f"actual_shape={tuple(masks.shape)}; actual_dtype={masks.dtype}"
                )
            self._validated_mask_address = masks.data_ptr()

        inputs = self._prepare_inputs(inputs_embeds)
        input_cache_slot = (
            self._flat_cache_slot_for_legacy_cache(legacy_cache)
            if flat_state is None
            else flat_state.slot
        )
        output_slot = self._next_cache_slot if input_cache_slot is None else 1 - input_cache_slot
        self._next_cache_slot = 1 - output_slot
        context_state = self._context_states[output_slot]
        context = context_state.context

        if flat_state is not None:
            if input_cache_slot not in (0, 1):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=flat-cache-state; "
                    f"expected=slot-0-or-1; actual={input_cache_slot}"
                )
            flat_buffer = self._flat_cache_buffer_slots[input_cache_slot]
            if (
                flat_buffer is None
                or flat_buffer.data_ptr() != flat_state.buffer.data_ptr()
            ):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=flat-cache-state; "
                    f"expected=registered-slot-buffer; slot={input_cache_slot}"
                )
            if flat_state.buffer.device != inputs.device or flat_state.buffer.dtype != self.dtype:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=flat-cache-state-device; "
                    f"expected={inputs.device}/{self.dtype}; "
                    f"actual={flat_state.buffer.device}/{flat_state.buffer.dtype}"
                )
            past_tokens = flat_state.length
            if not 1 <= past_tokens <= self.max_past_tokens:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-length; "
                    f"expected=1..{self.max_past_tokens}; actual={past_tokens}"
                )
            flat_input = _flat_token_major_view(
                flat_state.buffer,
                length=past_tokens,
                channels=self._flat_cache_channels(),
                head_dim=self.head_dim,
            )
            input_cache_key = ("slot", input_cache_slot)
        elif input_cache_slot is None:
            flat_input, past_tokens = self._pack_eager_cache_to_flat(
                legacy_cache, device=inputs.device
            )
            input_cache_key = ("ptr", flat_input.data_ptr())
        else:
            if legacy_cache is None:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-layout; "
                    "expected=legacy cache for pooled cache slot"
                )
            flat_buffer = self._flat_cache_buffer_slots[input_cache_slot]
            if flat_buffer is None:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=flat-cache-slot; "
                    f"slot={input_cache_slot}; expected=allocated-buffer"
                )
            pair = legacy_cache[0]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-layout; "
                    "expected=key-value-pair on flat pooled fast path"
                )
            first_tensor = pair[0]
            if not isinstance(first_tensor, torch.Tensor) or not first_tensor.is_cuda:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-device; "
                    "expected=CUDA tensor on flat pooled fast path"
                )
            shape = tuple(first_tensor.shape)
            if (
                len(shape) != 4
                or shape[0] != 1
                or shape[1] != self.num_kv_heads
                or shape[3] != self.head_dim
            ):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-shape; "
                    f"actual={shape} on flat pooled fast path"
                )
            past_tokens = shape[2]
            if not 1 <= past_tokens <= self.max_past_tokens:
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=cache-length; "
                    f"expected=1..{self.max_past_tokens}; actual={past_tokens}"
                )
            flat_input = _flat_token_major_view(
                flat_buffer,
                length=past_tokens,
                channels=self._flat_cache_channels(),
                head_dim=self.head_dim,
            )
            input_cache_key = ("slot", input_cache_slot)

        flat_input_shape = tuple(flat_input.shape)
        if not context_state.fixed_input_shapes_set:
            if not context.set_input_shape("inputs_embeds", tuple(inputs.shape)):
                raise RuntimeError("WordVoice TensorRT failed to set inputs_embeds shape")
            context_state.fixed_input_shapes_set = True
        context.set_tensor_address("inputs_embeds", inputs.data_ptr())
        if context_state.last_cache_shape != flat_input_shape:
            if not context.set_input_shape("past_cache", flat_input_shape):
                raise RuntimeError(
                    "WordVoice TensorRT failed to set flat cache input shape; "
                    f"shape={flat_input_shape}"
                )
            context_state.last_cache_shape = flat_input_shape
        if context_state.bound_input_cache_key != input_cache_key:
            context.set_tensor_address("past_cache", flat_input.data_ptr())
            context_state.bound_input_cache_key = input_cache_key

        hidden = self._ensure_hidden_buffer_slot(output_slot, device=inputs.device)
        if not context_state.validated_output_shapes:
            hidden_shape = tuple(context.get_tensor_shape("hidden_state"))
            if hidden_shape != tuple(hidden.shape):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=output-shape; "
                    f"tensor=hidden_state; expected={tuple(hidden.shape)}; actual={hidden_shape}"
                )
        context.set_tensor_address("hidden_state", hidden.data_ptr())
        output_buffer = self._ensure_flat_cache_buffer_slot(
            output_slot, device=inputs.device
        )
        output_length = past_tokens + 1
        flat_output = _flat_token_major_view(
            output_buffer,
            length=output_length,
            channels=self._flat_cache_channels(),
            head_dim=self.head_dim,
        )
        if not context_state.validated_output_shapes:
            present_shape = tuple(context.get_tensor_shape("present_cache"))
            if present_shape != tuple(flat_output.shape):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=output-shape; "
                    "tensor=present_cache; "
                    f"expected={tuple(flat_output.shape)}; actual={present_shape}"
                )
        if not context_state.output_addresses_bound:
            context.set_tensor_address("present_cache", flat_output.data_ptr())
        context_state.output_addresses_bound = True
        context_state.validated_output_shapes = True

        self._execute_async(
            context,
            inputs,
            started=started,
            past_tokens=past_tokens,
        )

        return hidden, _FlatCacheState(
            buffer=output_buffer,
            slot=output_slot,
            length=output_length,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def reset_metrics(self) -> None:
        self._device_seconds = 0.0
        self._host_seconds = 0.0
        self._steps = 0
        # Callers reset only after generation or an explicit CUDA sync (the
        # eager-parity gate), so the reusable event pool can start at zero.
        self._timing_event_cursor = 0

    def consume_metrics(self) -> dict[str, Any]:
        self._flush_device_timing()
        metrics: dict[str, Any] = {
            "decoder_backend": "tensorrt",
            "native_decode_seconds": round(self._device_seconds, 3),
            "native_host_seconds": round(self._host_seconds, 3),
            "native_transfer_seconds": 0.0,
            "native_decode_steps": self._steps,
            "native_cache_buffer_pool_allocations": self._cache_buffer_allocations,
            "native_flat_cache_buffer_allocations": self._flat_cache_buffer_allocations,
            "native_hidden_buffer_pool_allocations": self._hidden_buffer_allocations,
            "native_input_buffer_allocations": self._input_buffer_allocations,
            "engine_sha256": self.manifest["engine_sha256"],
            "manifest_sha256": sha256(self.manifest_path),
            "tensorrt_version": self.manifest["tensorrt_version"],
            "cuda_compute_capability": self.manifest["cuda_compute_capability"],
            "parity": dict(self._parity or {}),
        }
        self.reset_metrics()
        return metrics
