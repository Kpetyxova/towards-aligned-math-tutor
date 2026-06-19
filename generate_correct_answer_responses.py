"""Generate and attach tutor responses for correct-answer examples.

This script supports three workflow steps:

1. `generate-bank`: ask an OpenAI model for a bank of generic positive tutor
   responses and save them as JSON.
2. `combine-bank`: create first-sentence/second-sentence combinations from a
   generated response bank.
3. `assign` / `assign-prm800k`: randomly assign responses from that bank to
   correct-answer examples and save JSON arrays.

Examples:
    python generate_correct_answer_responses.py assign-prm800k \
        --prm-root ../prm800k/prm800k \
        --responses data/positive_tutor_responses_combinations.json

    python generate_correct_answer_responses.py assign \
        --responses data/positive_tutor_responses_combinations.json \
        --dataset-jsonl data/mr_gsm8k.json \
        --output data/mr-gsm8k_correct_solution_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None


DEFAULT_MODEL = "gpt-5"
DEFAULT_SEED = 42
DEFAULT_RESPONSES_JSON = Path("data/positive_tutor_responses_combinations.json")
DEFAULT_GENERATED_RESPONSES_JSON = Path("data/positive_tutor_responses.json")
DEFAULT_COMBINED_RESPONSES_JSON = Path("data/positive_tutor_responses_combinations.json")

SYSTEM_PROMPT = "You are an expert, encouraging math tutor."
USER_PROMPT = (
    "Generate exactly 100 distinct one-line tutor responses suitable for cases where a student's solution is correct.\n"
    "Requirements:\n"
    "- Each response must be concise, natural-sounding, and encouraging.\n"
    "- Vary the structure; do NOT start most lines with the same word or pattern.\n"
    "- Avoid semicolons entirely.\n"
    "- Avoid overly specific instructions such as asking for diagrams, substitution methods, coding, graphs, etc.\n"
    "- Follow-up suggestions must be general and applicable across many math topics (e.g., ask if they'd like another problem, another challenge, a summary, questions, etc.).\n"
    "- Tone should vary: praise, curiosity, invitations to continue, quick check-ins, etc.\n"
    "- Number each line 1) through 100).\n"
    "- No bullet points, no quotes, no emojis, no extra commentary.\n"
    "- Output plain text with exactly ONE response per line and EXACTLY 100 lines.\n"
)


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines into the process environment if unset."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json_array(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


def strip_numbering(line: str) -> str:
    return re.sub(r"^\s*(?:\d{1,3}[)\].:~-]?\s*|[-*]\s+)?", "", line).strip()


def load_tutor_responses(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        responses = json.load(fh)
    if not isinstance(responses, list) or not all(isinstance(item, str) for item in responses):
        raise ValueError(f"Expected a JSON list of strings in {path}")
    if not responses:
        raise ValueError(f"No tutor responses found in {path}")
    return responses


def save_tutor_responses(path: Path, responses: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(responses, fh, ensure_ascii=False, indent=2)


def generate_response_bank(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY, pass --api-key, or provide an .env file.")
    if OpenAI is None:
        raise SystemExit("Install the `openai` package to generate tutor responses.")

    client = OpenAI(api_key=api_key)
    completion = client.responses.create(
        model=args.model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        reasoning={"effort": "low"} if args.model.startswith("gpt-5") else None,
    )

    raw_text = getattr(completion, "output_text", "") or ""
    seen: set[str] = set()
    responses: list[str] = []
    for line in raw_text.splitlines():
        text = strip_numbering(line)
        if text and text not in seen:
            seen.add(text)
            responses.append(text)

    save_tutor_responses(args.output, responses)
    print(f"Saved {len(responses)} unnumbered responses to {args.output}")


def combine_response_bank(args: argparse.Namespace) -> None:
    prompts = load_tutor_responses(args.input)
    sentence_boundary_pattern = re.compile(r"(?<=[.!?])\s+")
    first_sentences: list[str] = []
    second_sentences: list[str] = []
    skipped: list[str] = []

    for prompt in prompts:
        parts = sentence_boundary_pattern.split(prompt.strip())
        if len(parts) < 2:
            skipped.append(prompt)
            continue
        first_sentences.append(parts[0].strip())
        second_sentences.append(parts[1].strip())

    combinations = [f"{first} {second}" for first, second in product(first_sentences, second_sentences)]
    save_tutor_responses(args.output, combinations)
    print(
        f"Saved {len(combinations)} combined responses to {args.output} "
        f"from {len(first_sentences)} first sentences and {len(second_sentences)} second sentences."
    )
    if skipped:
        print(f"Skipped {len(skipped)} responses without at least two sentences.")


def assign_responses(
    *,
    dataset_jsonl: Path,
    output: Path,
    responses: list[str],
    rng: random.Random,
    finish_reason: Optional[str],
) -> int:
    finish_reason = finish_reason.lower() if finish_reason else None
    records: list[dict[str, Any]] = []
    for record in read_jsonl(dataset_jsonl):
        if finish_reason:
            label = record.get("label") if isinstance(record.get("label"), dict) else {}
            actual_finish_reason = str(label.get("finish_reason", "")).lower()
            if actual_finish_reason != finish_reason:
                continue
        record["tutor_response"] = rng.choice(responses)
        records.append(record)

    write_json_array(output, records)
    return len(records)


def assign_single_dataset(args: argparse.Namespace) -> None:
    responses = load_tutor_responses(args.responses)
    rng = random.Random(args.seed)
    count = assign_responses(
        dataset_jsonl=args.dataset_jsonl,
        output=args.output,
        responses=responses,
        rng=rng,
        finish_reason=args.finish_reason,
    )
    print(f"Saved {count} filtered instances with tutor_response to {args.output}")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def split_sort_key(value: str) -> tuple[int, list[Any]]:
    preferred_order = {"train": 0, "test": 1}
    return (preferred_order.get(value, 2), natural_sort_key(value))


def discover_splits(prm_root: Path) -> list[str]:
    data_dir = prm_root / "data"
    splits = set()
    for path in data_dir.glob("phase*.filtered_by_math_splits_*.jsonl"):
        match = re.fullmatch(r"phase[^_]+_(.+)\.filtered_by_math_splits_.+\.jsonl", path.name)
        if match:
            splits.add(match.group(1))
    return sorted(splits, key=split_sort_key)


def discover_phases(prm_root: Path, splits: list[str]) -> list[str]:
    data_dir = prm_root / "data"
    split_set = set(splits)
    phases = set()
    for path in data_dir.glob("phase*.filtered_by_math_splits_*.jsonl"):
        match = re.fullmatch(r"(phase[^_]+)_(.+)\.filtered_by_math_splits_.+\.jsonl", path.name)
        if match and match.group(2) in split_set:
            phases.add(match.group(1))
    return sorted(phases, key=natural_sort_key)


def resolve_requested_splits(args: argparse.Namespace) -> list[str]:
    requested = args.splits
    if "all" in requested:
        if len(requested) > 1:
            raise SystemExit("Use --splits all by itself, or list concrete splits.")
        splits = discover_splits(args.prm_root)
    else:
        splits = requested
    if not splits:
        raise SystemExit(f"No filtered PRM800K split files found under {args.prm_root / 'data'}")
    return splits


def resolve_requested_phases(args: argparse.Namespace, splits: list[str]) -> list[str]:
    requested = args.phases
    if "all" in requested:
        if len(requested) > 1:
            raise SystemExit("Use --phases all by itself, or list concrete phases.")
        phases = discover_phases(args.prm_root, splits)
    else:
        phases = requested
    if not phases:
        raise SystemExit(f"No filtered PRM800K phase files found under {args.prm_root / 'data'}")
    return phases


def filtered_prm_path(prm_root: Path, phase: str, split: str) -> Path:
    return prm_root / "data" / f"{phase}_{split}.filtered_by_math_splits_{split}.jsonl"


def default_prm_output_path(dataset_jsonl: Path) -> Path:
    prefix = dataset_jsonl.name.split(".filtered_by_math_splits_", 1)[0]
    return dataset_jsonl.with_name(f"{prefix}.correct_solutions_responses.json")


def assign_prm800k(args: argparse.Namespace) -> None:
    responses = load_tutor_responses(args.responses)
    rng = random.Random(args.seed)
    splits = resolve_requested_splits(args)
    phases = resolve_requested_phases(args, splits)

    total = 0
    for split in splits:
        for phase in phases:
            dataset_jsonl = filtered_prm_path(args.prm_root, phase, split)
            if not dataset_jsonl.exists():
                raise SystemExit(f"Filtered PRM800K file does not exist: {dataset_jsonl}")
            output = args.output_dir / default_prm_output_path(dataset_jsonl).name if args.output_dir else default_prm_output_path(dataset_jsonl)
            count = assign_responses(
                dataset_jsonl=dataset_jsonl,
                output=output,
                responses=responses,
                rng=rng,
                finish_reason="solution",
            )
            total += count
            print(f"Saved {count} filtered instances with tutor_response to {output}")
    print(f"Saved {total} total PRM800K correct-answer instances.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-bank", help="Generate positive tutor response bank.")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_GENERATED_RESPONSES_JSON)
    generate_parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    generate_parser.add_argument("--api-key", default=None)
    generate_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    generate_parser.set_defaults(func=generate_response_bank)

    combine_parser = subparsers.add_parser("combine-bank", help="Create positive tutor response combinations.")
    combine_parser.add_argument("--input", type=Path, default=DEFAULT_GENERATED_RESPONSES_JSON)
    combine_parser.add_argument("--output", type=Path, default=DEFAULT_COMBINED_RESPONSES_JSON)
    combine_parser.set_defaults(func=combine_response_bank)

    assign_parser = subparsers.add_parser("assign", help="Assign tutor responses to one JSONL dataset.")
    assign_parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_JSON)
    assign_parser.add_argument("--dataset-jsonl", type=Path, required=True)
    assign_parser.add_argument("--output", type=Path, required=True)
    assign_parser.add_argument(
        "--finish-reason",
        default=None,
        help="Optional label.finish_reason value to keep, e.g. 'solution'.",
    )
    assign_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    assign_parser.set_defaults(func=assign_single_dataset)

    prm_parser = subparsers.add_parser("assign-prm800k", help="Assign responses to filtered PRM800K phase files.")
    prm_parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_JSON)
    prm_parser.add_argument("--prm-root", type=Path, default=Path("../prm800k/prm800k"))
    prm_parser.add_argument("--splits", type=parse_csv, default=["all"])
    prm_parser.add_argument("--phases", type=parse_csv, default=["all"])
    prm_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for outputs. Defaults to each input file's directory.",
    )
    prm_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prm_parser.set_defaults(func=assign_prm800k)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
