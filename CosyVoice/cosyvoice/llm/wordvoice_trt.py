"""Fail-closed TensorRT decode-step adapter for WordVoice's Qwen2 decoder."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


MANIFEST_SCHEMA = "wordvoice.tensorrt.decoder.v1"
ENGINE_FILENAME = "decoder.plan"
MANIFEST_FILENAME = "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_tensor_name(kind: str, layer: int, *, present: bool = False) -> str:
    prefix = "present" if present else "past"
    return f"{prefix}_{kind}_{layer}"


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
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=context-create; "
                f"engine={self.engine_path}"
            )
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

        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._execution_lock = threading.Lock()
        self._host_seconds = 0.0
        self._steps = 0
        self._validated_mask_address: int | None = None
        self._parity: dict[str, float] | None = None
        self._fixed_input_shapes_set = False
        self._last_cache_shape: tuple[int, ...] | None = None
        self._validated_output_shapes = False

    def _positive_int(self, name: str) -> int:
        value = self.manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=manifest-value; "
                f"field={name}; expected=positive-integer; actual={value!r}"
            )
        return value

    def _expected_tensor_names(self) -> set[str]:
        names = {"inputs_embeds", "hidden_state"}
        for layer in range(self.num_layers):
            for kind in ("key", "value"):
                names.add(cache_tensor_name(kind, layer))
                names.add(cache_tensor_name(kind, layer, present=True))
        return names

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
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Run the single shared TensorRT context without overlapping calls."""
        with self._execution_lock:
            return self._forward_one_step(inputs_embeds, masks, cache)

    def _forward_one_step(
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
        inputs = inputs_embeds.to(dtype=self.dtype).contiguous()
        cache_inputs: list[torch.Tensor] = []
        past_tokens: int | None = None
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
                cache_inputs.append(tensor.to(dtype=self.dtype).contiguous())
        if past_tokens is None or not 1 <= past_tokens <= self.max_past_tokens:
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=cache-length; "
                f"expected=1..{self.max_past_tokens}; actual={past_tokens}"
            )

        if not self._fixed_input_shapes_set:
            # The token input has a fixed shape for this batch-one decoder.
            # TensorRT retains it for the lifetime of the execution context.
            if not self.context.set_input_shape("inputs_embeds", tuple(inputs.shape)):
                raise RuntimeError("WordVoice TensorRT failed to set inputs_embeds shape")
            self._fixed_input_shapes_set = True
        self.context.set_tensor_address("inputs_embeds", inputs.data_ptr())
        cache_shape = tuple(cache_inputs[0].shape)
        if self._last_cache_shape != cache_shape:
            # Cache length changes only when a new token is appended. Repeated
            # lengths (for example across lines) do not need another shape
            # negotiation; tensor addresses are still refreshed every step.
            index = 0
            for layer in range(self.num_layers):
                for kind in ("key", "value"):
                    tensor = cache_inputs[index]
                    name = cache_tensor_name(kind, layer)
                    if not self.context.set_input_shape(name, tuple(tensor.shape)):
                        raise RuntimeError(
                            "WordVoice TensorRT failed to set cache input shape; "
                            f"tensor={name}; shape={tuple(tensor.shape)}"
                        )
                    index += 1
            self._last_cache_shape = cache_shape
        index = 0
        for layer in range(self.num_layers):
            for kind in ("key", "value"):
                self.context.set_tensor_address(
                    cache_tensor_name(kind, layer), cache_inputs[index].data_ptr()
                )
                index += 1

        hidden = torch.empty(
            (1, 1, self.hidden_size), device=inputs.device, dtype=self.dtype
        )
        # Present shapes are deterministic from the negotiated cache length.
        # Validate the first execution, then keep the hot path free of 49 host
        # shape queries per token while retaining address and execute checks.
        if not self._validated_output_shapes:
            hidden_shape = tuple(self.context.get_tensor_shape("hidden_state"))
            if hidden_shape != tuple(hidden.shape):
                raise RuntimeError(
                    "WordVoice TensorRT gate failed; stage=output-shape; "
                    f"tensor=hidden_state; expected={tuple(hidden.shape)}; actual={hidden_shape}"
                )
        self.context.set_tensor_address("hidden_state", hidden.data_ptr())
        present: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.num_layers):
            pair: list[torch.Tensor] = []
            for kind in ("key", "value"):
                tensor = torch.empty(
                    (1, self.num_kv_heads, past_tokens + 1, self.head_dim),
                    device=inputs.device,
                    dtype=self.dtype,
                )
                name = cache_tensor_name(kind, layer, present=True)
                if not self._validated_output_shapes:
                    actual_shape = tuple(self.context.get_tensor_shape(name))
                    if actual_shape != tuple(tensor.shape):
                        raise RuntimeError(
                            "WordVoice TensorRT gate failed; stage=output-shape; "
                            f"tensor={name}; expected={tuple(tensor.shape)}; "
                            f"actual={actual_shape}"
                        )
                self.context.set_tensor_address(
                    name, tensor.data_ptr()
                )
                pair.append(tensor)
            present.append((pair[0], pair[1]))
        self._validated_output_shapes = True

        current_stream = torch.cuda.current_stream(inputs.device)
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record(current_stream)
        if not self.context.execute_async_v3(current_stream.cuda_stream):
            raise RuntimeError(
                "WordVoice TensorRT gate failed; stage=execute; "
                f"engine={self.engine_path}; past_tokens={past_tokens}"
            )
        event_end.record(current_stream)
        self._events.append((event_start, event_end))
        self._host_seconds += time.perf_counter() - started
        self._steps += 1
        return hidden, tuple(present)

    def reset_metrics(self) -> None:
        self._events.clear()
        self._host_seconds = 0.0
        self._steps = 0

    def consume_metrics(self) -> dict[str, Any]:
        if self._events:
            self._events[-1][1].synchronize()
        device_seconds = sum(
            start.elapsed_time(end) / 1000.0 for start, end in self._events
        )
        metrics: dict[str, Any] = {
            "decoder_backend": "tensorrt",
            "native_decode_seconds": round(device_seconds, 3),
            "native_host_seconds": round(self._host_seconds, 3),
            "native_transfer_seconds": 0.0,
            "native_decode_steps": self._steps,
            "engine_sha256": self.manifest["engine_sha256"],
            "manifest_sha256": sha256(self.manifest_path),
            "tensorrt_version": self.manifest["tensorrt_version"],
            "cuda_compute_capability": self.manifest["cuda_compute_capability"],
            "parity": dict(self._parity or {}),
        }
        self.reset_metrics()
        return metrics
