#!/usr/bin/env python3
"""Run inference for the aligned math tutor on dialog feedback data.

The default model source is the Hugging Face PEFT adapter:

    kpetyxova/Qwen3-8B-aligned-math-tutor-lora

You can also pass a local adapter folder, or a merged/full model folder, with
`--model-source`. Prompt construction matches `train.py` and
supports the same v1-v4 input variants.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from train import INPUT_VAR_DESCRIPTIONS, build_feedback_sample, parse_input_var


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "training_sets" / "dialog_data_ft" / "dialog_data_ft_test.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inference"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MODEL_SOURCE = "kpetyxova/Qwen3-8B-aligned-math-tutor-lora"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object in {path} at line {line_number}")
        rows.append(row)
    return rows


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def build_prompt(sample: dict[str, Any], *, input_var: str) -> str:
    is_correct = coerce_bool(sample.get("is_correct"), default=False)
    return build_feedback_sample(sample, is_correct, input_var=input_var)["prompt"]


def slugify(value: str) -> str:
    value = value.strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def default_output_path(args: argparse.Namespace) -> Path:
    source_slug = slugify(args.model_source)
    data_slug = Path(args.data_path).stem
    return resolve_path(args.output_dir) / f"{source_slug}_{args.input_var}_{data_slug}.jsonl"


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def import_inference_dependencies() -> SimpleNamespace:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install torch, transformers, and peft before running inference.") from exc

    return SimpleNamespace(
        torch=torch,
        PeftModel=PeftModel,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )


def get_torch_dtype(args: argparse.Namespace, deps: SimpleNamespace) -> Any:
    if args.torch_dtype == "bf16":
        return deps.torch.bfloat16
    if args.torch_dtype == "fp16":
        return deps.torch.float16
    if args.torch_dtype == "fp32":
        return deps.torch.float32
    if deps.torch.cuda.is_available():
        return deps.torch.bfloat16 if deps.torch.cuda.is_bf16_supported() else deps.torch.float16
    return deps.torch.float32


def local_model_kind(source: str) -> str | None:
    path = Path(source).expanduser()
    if not path.exists():
        return None
    if (path / "adapter_config.json").exists():
        return "adapter"
    if (path / "config.json").exists():
        return "full"
    return None


def infer_model_kind(args: argparse.Namespace) -> str:
    if args.model_kind != "auto":
        return args.model_kind
    local_kind = local_model_kind(args.model_source)
    if local_kind:
        return local_kind
    return "adapter"


def load_tokenizer(source: str, fallback: str, deps: SimpleNamespace, *, trust_remote_code: bool) -> Any:
    kwargs = {"trust_remote_code": trust_remote_code}
    try:
        tokenizer = deps.AutoTokenizer.from_pretrained(source, use_fast=False, **kwargs)
    except Exception:
        try:
            tokenizer = deps.AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
        except Exception:
            try:
                tokenizer = deps.AutoTokenizer.from_pretrained(fallback, use_fast=False, **kwargs)
            except Exception:
                tokenizer = deps.AutoTokenizer.from_pretrained(fallback, use_fast=True, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_tokenizer(args: argparse.Namespace, deps: SimpleNamespace) -> tuple[Any, Any, str]:
    model_kind = infer_model_kind(args)
    dtype = get_torch_dtype(args, deps)
    common_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }

    if model_kind == "full":
        model = deps.AutoModelForCausalLM.from_pretrained(args.model_source, **common_kwargs)
        tokenizer = load_tokenizer(
            args.model_source,
            args.base_model,
            deps,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        base = deps.AutoModelForCausalLM.from_pretrained(args.base_model, **common_kwargs)
        model = deps.PeftModel.from_pretrained(base, args.model_source)
        tokenizer = load_tokenizer(
            args.model_source,
            args.base_model,
            deps,
            trust_remote_code=args.trust_remote_code,
        )

    model.eval()
    return model, tokenizer, model_kind


def get_input_device(model: Any, deps: SimpleNamespace) -> Any:
    if hasattr(model, "device"):
        device = model.device
        if getattr(device, "type", None) != "meta":
            return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return deps.torch.device("cuda" if deps.torch.cuda.is_available() else "cpu")


def generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
    deps: SimpleNamespace,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
    )
    device = get_input_device(model, deps)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    if args.do_sample:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        generation_kwargs["do_sample"] = False

    with deps.torch.no_grad():
        sequences = model.generate(**inputs, **generation_kwargs)

    prompt_width = inputs["input_ids"].shape[1]
    generated = sequences[:, prompt_width:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def write_inference_outputs(args: argparse.Namespace, output_path: Path) -> None:
    data_path = resolve_path(args.data_path)
    rows = read_json_or_jsonl(data_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found in {data_path}")

    print(f"Data: {data_path}")
    print(f"Rows: {len(rows)}")
    print(f"Input variant: {args.input_var} ({INPUT_VAR_DESCRIPTIONS[args.input_var]})")
    print(f"Model source: {args.model_source}")
    print(f"Output: {output_path}")

    if args.dry_run:
        for index, row in enumerate(rows[: min(2, len(rows))]):
            prompt = build_prompt(row, input_var=args.input_var)
            print(f"\n--- Prompt preview {index} ---")
            print(prompt[:2000])
            print(f"\nGold tutor response: {str(row.get('tutor_response', ''))[:500]}")
        print("\nDry run complete. No model was loaded.")
        return

    deps = import_inference_dependencies()
    model, tokenizer, model_kind = load_model_and_tokenizer(args, deps)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = lambda x, **_kwargs: x  # noqa: E731

    with output_path.open("w", encoding="utf-8") as fh:
        global_index = 0
        total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
        for batch in tqdm(batched(rows, args.batch_size), total=total_batches, desc="Generating"):
            prompts = [build_prompt(row, input_var=args.input_var) for row in batch]
            outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                args=args,
                deps=deps,
            )
            for row, prompt, output in zip(batch, prompts, outputs):
                record = {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "is_correct": coerce_bool(row.get("is_correct"), default=False),
                    "input_var": args.input_var,
                    "base_model": args.base_model,
                    "model_source": args.model_source,
                    "model_kind": model_kind,
                    "dialog_history": row.get("dialog_history", ""),
                    "gold_solution": row.get("gold_solution", ""),
                    "gold_tutor_response": row.get("tutor_response", ""),
                    "prompt": prompt,
                    "output": output,
                    "row_index": global_index,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_index += 1
            fh.flush()

    print(f"Wrote {len(rows)} generations to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference on dialog_data_ft with a HF or local aligned math tutor model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE, help="HF repo id or local model/adapter folder.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model used when --model-source is a PEFT adapter.")
    parser.add_argument("--model-kind", choices=["auto", "adapter", "full"], default="auto")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-path")
    parser.add_argument(
        "--input-var",
        "--input_var",
        "--input-variant",
        "--input_variant",
        dest="input_var",
        type=parse_input_var,
        default="v4",
        help="Prompt input variant: v1 dialog only, v2 dialog + correctness flag, v3 dialog + gold solution, v4 dialog + both.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--greedy", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=True)

    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = resolve_path(args.output_path) if args.output_path else default_output_path(args)
    write_inference_outputs(args, output_path)


if __name__ == "__main__":
    main()
