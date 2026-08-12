"""Create one exact WordVoice request bundle for native MLX evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice-model", required=True)
    parser.add_argument("--aligner-model", required=True)
    parser.add_argument("--llm-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--hyper-yaml", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--controls-json", default="{}")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import wordvoice_infer
    from cosyvoice.cli.cosyvoice import WordVoice
    from get_timestamp_mmsfa import MMSFA_Aligner

    controls = json.loads(args.controls_json)
    if not isinstance(controls, dict):
        raise TypeError("--controls-json must decode to an object")
    wordvoice_infer.Aligner_Model = MMSFA_Aligner(model_path=args.aligner_model)
    wordvoice_infer.wordvoice = WordVoice(
        model_dir=args.cosyvoice_model,
        llm_path=args.llm_checkpoint,
        flow_path=args.flow_checkpoint,
        hyper_yaml_path=args.hyper_yaml,
    )
    metrics = wordvoice_infer.eval(
        args.prompt_text,
        args.prompt_audio,
        args.text,
        controls,
        str(args.output.with_suffix(".unused.wav")),
        args.language,
        prepared_request_path=args.output,
        export_only=True,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
