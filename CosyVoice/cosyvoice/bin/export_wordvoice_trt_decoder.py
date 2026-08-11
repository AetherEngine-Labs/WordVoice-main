"""Export and build the platform-specific TensorRT decoder for WordVoice."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from cosyvoice.llm.wordvoice_trt import (
    ENGINE_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    cache_tensor_name,
    sha256,
)


class Qwen2DecodeStep(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_layers: int, precision: str):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.precision = precision

    def forward(self, inputs_embeds: torch.Tensor, *flat_cache: torch.Tensor):
        legacy_cache = tuple(
            (flat_cache[index], flat_cache[index + 1])
            for index in range(0, len(flat_cache), 2)
        )
        dynamic_cache = DynamicCache.from_legacy_cache(legacy_cache)
        precision_context = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.precision == "autocast_fp16"
            else nullcontext()
        )
        with precision_context:
            output = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=dynamic_cache,
                use_cache=True,
                return_dict=True,
            )
        cache = output.past_key_values
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        flattened = tuple(tensor for pair in cache for tensor in pair)
        return (output.last_hidden_state[:, -1:, :], *flattened)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_markers() -> None:
    if not os.environ.get("AGENT_OWNER") or not os.environ.get("AGENT_TASK"):
        raise RuntimeError("AGENT_OWNER and AGENT_TASK are required")


def load_wordvoice_qwen(
    base_model_dir: Path, checkpoint: Path, precision: str
) -> torch.nn.Module:
    model = Qwen2ForCausalLM.from_pretrained(
        str(base_model_dir), torch_dtype=torch.float32
    ).model
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    prefix = "llm.model.model."
    model_state = {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "WordVoice TensorRT build gate failed; stage=checkpoint-load; "
            f"missing={missing}; unexpected={unexpected}"
        )
    model.config._attn_implementation = "eager"
    if precision == "autocast_fp16":
        return model.eval().float().cuda()
    return model.eval().float().cuda()


def export_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    precision: str,
) -> None:
    wrapper = Qwen2DecodeStep(model, num_layers, precision).eval()
    tensor_dtype = torch.float16 if precision == "autocast_fp16" else torch.float32
    inputs = [
        torch.linspace(
            -0.1,
            0.1,
            steps=hidden_size,
            device="cuda",
            dtype=tensor_dtype,
        ).reshape(1, 1, hidden_size)
    ]
    for _ in range(num_layers):
        for _kind in ("key", "value"):
            inputs.append(
                torch.zeros(
                    1,
                    num_kv_heads,
                    32,
                    head_dim,
                    device="cuda",
                    dtype=tensor_dtype,
                )
            )
    input_names = ["inputs_embeds"] + [
        cache_tensor_name(kind, layer)
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]
    output_names = ["hidden_state"] + [
        cache_tensor_name(kind, layer, present=True)
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]
    dynamic_axes = {
        name: {2: "past_tokens"} for name in input_names if name != "inputs_embeds"
    }
    dynamic_axes.update(
        {name: {2: "present_tokens"} for name in output_names if name != "hidden_state"}
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            str(onnx_path),
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    max_past_tokens: int,
    precision: str,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    builder = trt.Builder(logger)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(
            "WordVoice TensorRT build gate failed; stage=onnx-parse; "
            f"errors={errors}; onnx={onnx_path}"
        )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "inputs_embeds",
        (1, 1, hidden_size),
        (1, 1, hidden_size),
        (1, 1, hidden_size),
    )
    for layer in range(num_layers):
        for kind in ("key", "value"):
            profile.set_shape(
                cache_tensor_name(kind, layer),
                (1, num_kv_heads, 1, head_dim),
                (1, num_kv_heads, 256, head_dim),
                (1, num_kv_heads, max_past_tokens, head_dim),
            )
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "WordVoice TensorRT build gate failed; stage=engine-build; "
            f"onnx={onnx_path}"
        )
    engine_path.write_bytes(bytes(serialized))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--wordvoice-llm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-past-tokens", type=int, default=2048)
    parser.add_argument(
        "--precision", choices=("fp32", "autocast_fp16"), required=True
    )
    parser.add_argument("--replace-rejected-engine", action="store_true")
    parser.add_argument("--remove-rejected-staging-only", action="store_true")
    parser.add_argument("--agent-owner", required=True)
    parser.add_argument("--agent-task", required=True)
    args = parser.parse_args()
    require_markers()
    if (
        args.agent_owner != os.environ["AGENT_OWNER"]
        or args.agent_task != os.environ["AGENT_TASK"]
    ):
        raise RuntimeError(
            "process marker arguments must match AGENT_OWNER and AGENT_TASK"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("WordVoice TensorRT build requires CUDA")
    if args.max_past_tokens < 512:
        raise RuntimeError("--max-past-tokens must be at least 512")
    for required in (args.base_model_dir, args.wordvoice_llm):
        if not required.exists():
            raise RuntimeError(f"WordVoice TensorRT build input is missing: {required}")
    if args.remove_rejected_staging_only:
        manifest_path = args.output_dir / MANIFEST_FILENAME
        rejection_path = (
            args.output_dir
            / "failed-attempts"
            / "003-autocast-fp16-nonfinite-rejected.json"
        )
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
        prior_onnx = args.output_dir / "decoder.onnx"
        prior_engine = args.output_dir / ENGINE_FILENAME
        if (
            prior_manifest.get("engine_sha256") != rejection.get("engine_sha256")
            or prior_manifest.get("onnx_sha256") != rejection.get("onnx_sha256")
            or sha256(prior_engine) != rejection.get("engine_sha256")
            or sha256(prior_onnx) != rejection.get("onnx_sha256")
        ):
            raise RuntimeError(
                "WordVoice TensorRT rejected-staging cleanup gate failed"
            )
        removed = {
            "schema": "wordvoice.tensorrt.rejected-staging-removal.v1",
            "engine_sha256": rejection["engine_sha256"],
            "onnx_sha256": rejection["onnx_sha256"],
            "engine_bytes": prior_engine.stat().st_size,
            "onnx_bytes": prior_onnx.stat().st_size,
        }
        prior_engine.unlink()
        prior_onnx.unlink()
        manifest_path.unlink()
        atomic_json(
            args.output_dir / "failed-attempts" / "004-staging-removed.json",
            removed,
        )
        print(json.dumps(removed, indent=2))
        return 0
    if args.output_dir.exists() and args.replace_rejected_engine:
        manifest_path = args.output_dir / MANIFEST_FILENAME
        rejection_path = (
            args.output_dir
            / "failed-attempts"
            / "002-all-fp16-parity-rejected.json"
        )
        if not manifest_path.is_file() or not rejection_path.is_file():
            raise RuntimeError(
                "WordVoice TensorRT replacement gate requires the prior manifest "
                f"and rejection receipt; output={args.output_dir}"
            )
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
        prior_onnx = args.output_dir / "decoder.onnx"
        prior_engine = args.output_dir / ENGINE_FILENAME
        if (
            prior_manifest.get("status") != "engine_built_parity_pending"
            or prior_manifest.get("engine_sha256") != rejection.get("engine_sha256")
            or prior_manifest.get("onnx_sha256") != rejection.get("onnx_sha256")
            or sha256(prior_engine) != rejection.get("engine_sha256")
            or sha256(prior_onnx) != rejection.get("onnx_sha256")
        ):
            raise RuntimeError(
                "WordVoice TensorRT replacement gate failed; prior hashes or status drifted"
            )
        prior_engine.unlink()
        prior_onnx.unlink()
        manifest_path.unlink()
    elif args.output_dir.exists():
        entries = list(args.output_dir.iterdir())
        failure_path = args.output_dir / "failure.json"
        if entries != [failure_path]:
            raise RuntimeError(
                "WordVoice TensorRT build refuses to overwrite an existing directory; "
                f"output={args.output_dir}; entries={[path.name for path in entries]}"
            )
        attempts_dir = args.output_dir / "failed-attempts"
        attempts_dir.mkdir()
        failure_path.replace(attempts_dir / "001-legacy-cache-export.json")
    else:
        args.output_dir.mkdir(parents=True)
    onnx_path = args.output_dir / "decoder.onnx"
    engine_path = args.output_dir / ENGINE_FILENAME
    started = time.perf_counter()
    try:
        model = load_wordvoice_qwen(
            args.base_model_dir, args.wordvoice_llm, args.precision
        )
        config = model.config
        head_dim = config.hidden_size // config.num_attention_heads
        export_started = time.perf_counter()
        export_onnx(
            model,
            onnx_path,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            precision=args.precision,
        )
        export_seconds = time.perf_counter() - export_started
        import onnx

        onnx.checker.check_model(str(onnx_path))
        build_started = time.perf_counter()
        build_engine(
            onnx_path,
            engine_path,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            max_past_tokens=args.max_past_tokens,
            precision=args.precision,
        )
        build_seconds = time.perf_counter() - build_started
        import tensorrt as trt

        capability = torch.cuda.get_device_capability()
        builder_source = Path(__file__).resolve()
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "engine_built_parity_pending",
            "precision": args.precision,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "tensorrt_version": trt.__version__,
            "gpu_name": torch.cuda.get_device_name(),
            "cuda_compute_capability": f"{capability[0]}.{capability[1]}",
            "num_hidden_layers": config.num_hidden_layers,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": head_dim,
            "hidden_size": config.hidden_size,
            "max_past_tokens": args.max_past_tokens,
            "base_model_config_sha256": sha256(args.base_model_dir / "config.json"),
            "wordvoice_llm_sha256": sha256(args.wordvoice_llm),
            "builder_source_sha256": sha256(builder_source),
            "onnx_sha256": sha256(onnx_path),
            "engine_sha256": sha256(engine_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "engine_bytes": engine_path.stat().st_size,
            "export_seconds": round(export_seconds, 3),
            "engine_build_seconds": round(build_seconds, 3),
            "total_build_seconds": round(time.perf_counter() - started, 3),
            "command": [sys.executable, "-m", "cosyvoice.bin.export_wordvoice_trt_decoder"],
        }
        atomic_json(args.output_dir / MANIFEST_FILENAME, manifest)
        print(json.dumps(manifest, indent=2))
    except Exception as error:
        failure = {
            "schema": MANIFEST_SCHEMA,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "total_build_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_json(args.output_dir / "failure.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
