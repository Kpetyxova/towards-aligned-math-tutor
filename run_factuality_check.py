"""Run GPT-5 factuality checks on `inference.py` JSONL outputs.

Input records should match the inference JSONL format:
each row has `dialog_history`, `gold_solution`, `is_correct`,
`gold_tutor_response`, and `output`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "inference"

PRE_CHECK_SYSTEM_PROMPT = """You are an expert pedagogical judge.
Your task is to evaluate a math tutor's response to a student based on factuality and categorize any factual issue.

**Factuality**: The response should not invent numbers, calculations, or details not present in the problem or the student's history.

Return fields:
- is_factual: true/false
- student_has_mistake: true/false
- tutor_acknowledged_no_mistake: true/false/null
- category: one of
    - student_hallucination: Tutor hallucinates something in the student's solution
    - task_hallucination: Tutor hallucinates something in task definition
    - response_incorrect: Tutor's feedback/solution is not factually correct
    - other: Explain what is wrong
  If is_factual is true, set category to null.
- reasoning: brief explanation

Be strict: if any factual issue exists, set is_factual to false and choose the best-fitting category.

Provide your evaluation in the strictly defined structure.
"""


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def load_dotenv_if_present(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def import_eval_dependencies() -> SimpleNamespace:
    try:
        from openai import OpenAI
        from pydantic import BaseModel, Field
    except Exception as exc:
        raise RuntimeError("Install openai and pydantic before running factuality checks.") from exc

    return SimpleNamespace(OpenAI=OpenAI, BaseModel=BaseModel, Field=Field)


def make_precheck_result_class(deps: SimpleNamespace) -> Any:
    class PreCheckResult(deps.BaseModel):
        is_factual: bool = deps.Field(..., description="Is the tutor's response factual and free of hallucinations?")
        student_has_mistake: bool = deps.Field(..., description="Does the student's last utterance contain a mistake?")
        tutor_acknowledged_no_mistake: Optional[bool] = deps.Field(
            None,
            description=(
                "If the student has no mistake, did the tutor acknowledge that? "
                "Only applicable if student_has_mistake is False."
            ),
        )
        category: Optional[str] = deps.Field(
            None,
            description=(
                "If not factual, classify the issue as one of: "
                "'student_hallucination', 'task_hallucination', 'response_incorrect', or 'other'."
            ),
        )
        reasoning: str = deps.Field(..., description="Brief explanation for the decisions.")

    return PreCheckResult


def model_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result.dict()


def make_user_prompt(record: dict[str, Any], *, response_key: str) -> str:
    gold_tutor_response = record.get("gold_tutor_response", record.get("tutor_response", ""))
    return f"""
[Dialog History]
{record.get("dialog_history", "")}

[Gold Solution to the Task]
{record.get("gold_solution", "")}

[Is student's solution correct?]
{coerce_bool(record.get("is_correct"), default=False)}

[Gold Tutor Response]
{gold_tutor_response}

[Generated Response to Evaluate]
{record.get(response_key, "")}
"""


def parse_completion(client: Any, *, model: str, messages: list[dict[str, str]], response_format: Any) -> Any:
    if hasattr(client.chat.completions, "parse"):
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        return completion.choices[0].message.parsed

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    return completion.choices[0].message.parsed


def factuality_check(
    client: Any,
    *,
    model: str,
    record: dict[str, Any],
    response_key: str,
    precheck_result_cls: Any,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": PRE_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(record, response_key=response_key)},
    ]

    for attempt in range(max_retries):
        try:
            result = parse_completion(
                client,
                model=model,
                messages=messages,
                response_format=precheck_result_cls,
            )
            annotation = model_to_dict(result)
            annotation["judge_model"] = model
            return annotation
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {
                "is_factual": False,
                "student_has_mistake": True,
                "tutor_acknowledged_no_mistake": None,
                "category": "other",
                "reasoning": f"Error during factuality check: {exc}",
                "judge_model": model,
                "error": str(exc),
            }

    raise RuntimeError("Unreachable factuality check retry state.")


def stable_record_key(record: dict[str, Any]) -> str:
    for key in ("row_index", "id"):
        if key in record:
            return f"{key}:{record[key]}"
    return json.dumps(
        {
            "dialog_history": record.get("dialog_history", ""),
            "output": record.get("output", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def load_completed_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    for row in read_json_or_jsonl(output_path):
        completed.add(stable_record_key(row))
    return completed


def default_output_path(input_path: Path, model: str) -> Path:
    model_slug = model.replace("/", "_")
    return input_path.with_name(f"{input_path.stem}_factuality_{model_slug}.jsonl")


def run(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input_path)
    output_path = resolve_path(args.output_path) if args.output_path else default_output_path(input_path, args.model)
    records = read_json_or_jsonl(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records found in {input_path}")

    missing_response = sum(not record.get(args.response_key) for record in records)
    if missing_response:
        raise ValueError(f"{missing_response} records are missing response key {args.response_key!r}.")

    print(f"Input: {input_path}")
    print(f"Rows: {len(records)}")
    print(f"Response key: {args.response_key}")
    print(f"Judge model: {args.model}")
    print(f"Output: {output_path}")

    if args.dry_run:
        preview = make_user_prompt(records[0], response_key=args.response_key)
        print("\n--- System Prompt ---")
        print(PRE_CHECK_SYSTEM_PROMPT)
        print("\n--- User Prompt Preview ---")
        print(preview[:3000])
        print("\nDry run complete. No API call was made.")
        return

    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --resume or --overwrite.")
    if args.overwrite and output_path.exists():
        output_path.unlink()

    load_dotenv_if_present([SCRIPT_DIR / ".env", SCRIPT_DIR.parent / ".env", Path.cwd() / ".env"])
    deps = import_eval_dependencies()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    client = deps.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    precheck_result_cls = make_precheck_result_class(deps)
    completed = load_completed_keys(output_path) if args.resume else set()

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = lambda x, **_kwargs: x  # noqa: E731

    processed = 0
    skipped = 0
    for record in tqdm(records, desc="Factuality checks"):
        if stable_record_key(record) in completed:
            skipped += 1
            continue
        annotation = factuality_check(
            client,
            model=args.model,
            record=record,
            response_key=args.response_key,
            precheck_result_cls=precheck_result_cls,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        output_record = dict(record)
        output_record.setdefault("llm_annotation", {})
        output_record["llm_annotation"]["factuality_check"] = annotation
        append_jsonl(output_path, output_record)
        processed += 1

    print(f"Wrote {processed} factuality annotations to {output_path}")
    if skipped:
        print(f"Skipped {skipped} already-annotated rows due to --resume.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only the GPT-5 factuality pre-check on inference JSONL outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to a JSONL/JSON file produced by inference.py.",
    )
    parser.add_argument("--output-path")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--response-key", default="output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
