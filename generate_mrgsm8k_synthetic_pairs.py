"""Generate RMBoost-style synthetic preference pairs for MR-GSM8K.

This script:
1. loads MR-GSM8K from Hugging Face (`Randolphzeng/Mr-GSM8K`) or a local JSONL file;
2. skips configured question types, such as `POT`;
3. keeps only samples whose model answer is marked wrong;
4. builds the student's attempt from `attempt` or `model_output_steps`;
5. generates one preferred tutor response and one degraded response per
   requested pedagogical aspect.

Example:
    python generate_mrgsm8k_synthetic_pairs.py \
        --output data/mrgsm8k_synthetic_pairs.jsonl \
        --limit 1 \
        --aspects Clarity

Set `OPENAI_API_KEY` before running generation. Use `--dry-run` to inspect the
prepared inputs without calling the API.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None


PEDAGOGICAL_ASPECTS = {
    "Factuality": """Factuality: the tutor response should be mathematically correct, avoid contradicting the student, and avoid irrelevant or unsupported claims. To degrade this aspect, introduce a subtle mathematical or contextual error while keeping the response plausible.""",
    "MistakeIdentification": """Mistake identification: the tutor response should make clear that the student's solution contains a mistake. To degrade this aspect, avoid identifying that anything is wrong, or respond as if the student's reasoning is acceptable.""",
    "Targetedness": """Targetedness: the tutor response should address the student's core misconception or erroneous step. To degrade this aspect, give generic advice or focus on a different part of the solution.""",
    "RevealingAnswer": """Not revealing the final answer: the tutor may give a substep when needed, but should avoid giving away the final answer. To degrade this aspect, reveal the final answer or enough of the final computation that the student no longer has meaningful work to do.""",
    "Clarity": """Clarity: the tutor response should be easy to follow and should connect the student's work to the next step. To degrade this aspect, keep the broad idea but make the wording vague, poorly connected, or hard to act on.""",
}

SCAFFOLD_LEVELS = {
    "L0": "Meta-cognitive - gentle, tone-first nudge with a brief reflection question, usually 1-2 short sentences.",
    "L1": "Strategic - high-level guidance and a possible next move, around 2-3 short sentences.",
    "L2": "Conceptual - point out the likely misunderstanding and highlight the key idea, about 3-5 concise lines.",
    "L3": "Step-by-step - locate the error and suggest a short concrete sequence of next steps.",
}

SYSTEM_TUTOR = "You are an expert math tutor who gives concise, actionable feedback."
SYSTEM_EDITOR = "You are revising feedback to control its quality along requested aspects."

Y_POS_PROMPT = """
You are an expert math tutor helping a student understand and fix a mistake in
their solution. Your goal is to guide the student toward the correct reasoning
without revealing the final answer. Keep your response concise and focused on
the core misunderstanding.

Good tutor responses are evaluated using these pedagogical criteria:
1. Factuality: be mathematically correct, do not contradict the student, and do
   not add irrelevant information.
2. Mistake Identification: make clear, explicitly or implicitly, that there is a
   mistake in the student's solution.
3. Targetedness: address the core misconception or misunderstanding.
4. Revealing Answer: avoid giving away the final answer, though a substep can be
   shared when necessary.
5. Clarity: avoid awkward, confusing, or misleading wording.

Assigned scaffolding level:
{level}: {level_notes}

Problem:
{question}

Student solution:
{attempt}

Gold solution for reference:
{correct_solution}

Known erroneous step:
{error_step}

Error reason:
{error_reason}

Write the tutor's next response. Follow the assigned scaffolding level, focus on
the student's first substantive mistake, and do not reveal the final answer.
Wrap the response in <response>...</response> tags.
"""

Y_NEG_PROMPT = """
You are an expert tutor teaching another tutor how not to respond to a student's
incorrect math solution.

Only degrade the following aspect:
{aspect}

Aspect-specific degradation instructions:
{aspect_instructions}

Problem:
{question}

Student solution:
{attempt}

Gold solution for reference:
{correct_solution}

Known erroneous step:
{error_step}

Error reason:
{error_reason}

The preferred tutor response is:
{good_response}

Revise the preferred tutor response so that it clearly fails on the requested
aspect while preserving as much of the original wording as possible. Do not
rewrite it from scratch unless needed. Put the degraded response in
<response>...</response> tags.
"""


@dataclass
class ModelInput:
    uuid: Optional[str]
    question: str
    attempt: str
    correct_solution: str
    error_step: str
    error_reason: str
    question_type: Optional[str] = None


@dataclass
class PreferencePair:
    uuid: Optional[str]
    question: str
    attempt: str
    correct_solution: str
    error_step: str
    error_reason: str
    y_pos: str
    y_neg: str
    targeted_aspects: list[str]
    level: str
    level_notes: str
    source: str = "mr-gsm8k"
    question_type: Optional[str] = None


class LLMError(RuntimeError):
    """Raised when an LLM request fails after retries."""


@dataclass
class OpenAIClient:
    model: str
    temperature: Optional[float] = 1.0
    max_retries: int = 3
    timeout: Optional[float] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise LLMError("Install the `openai` package to run generation.")
        self._client = OpenAI(
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "max_output_tokens": max_tokens,
        }
        if self.temperature is not None and not self.model.startswith("gpt-5"):
            payload["temperature"] = self.temperature
        if self.model.startswith("gpt-5"):
            payload["reasoning"] = {"effort": "low"}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.responses.create(**payload)
                text = getattr(response, "output_text", "") or ""
                return text.strip()
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(1.5 * attempt)
        raise LLMError(f"Failed to call OpenAI API after {self.max_retries} attempts: {last_error}")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_dataset_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input:
        yield from read_jsonl(args.input)
        return

    try:
        from datasets import DatasetDict, load_dataset
    except Exception:
        raise SystemExit("Install the `datasets` package or pass --input with a local JSONL file.")

    load_kwargs: dict[str, Any] = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    if args.dataset_split:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split, **load_kwargs)
        for record in dataset:
            yield dict(record)
        return

    dataset = load_dataset(args.dataset_name, **load_kwargs)
    if DatasetDict is not None and isinstance(dataset, DatasetDict):
        for split_name in dataset:
            for record in dataset[split_name]:
                item = dict(record)
                item.setdefault("dataset_split", split_name)
                yield item
    else:
        for record in dataset:
            yield dict(record)


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if value is not None and str(value).strip() else []


def to_model_input(record: dict[str, Any]) -> Optional[ModelInput]:
    question = str(record.get("question", "")).strip()
    correct_solution = str(record.get("ground_truth_solution") or record.get("correct_solution") or "").strip()
    attempt = str(record.get("attempt", "")).strip()
    if not attempt:
        attempt = "\n".join(parse_steps(record.get("model_output_steps", []))).strip()

    if not question or not attempt or not correct_solution:
        return None

    return ModelInput(
        uuid=str(record.get("uuid")).strip() if record.get("uuid") else None,
        question=question,
        attempt=attempt,
        correct_solution=correct_solution,
        error_step=str(record.get("model_output_solution_first_error_step") or record.get("error_step") or "Not available").strip(),
        error_reason=str(record.get("model_output_solution_first_error_reason") or record.get("error_reason") or "Not available").strip(),
        question_type=str(record.get("question_type")).strip() if record.get("question_type") else None,
    )


def strip_response_tags(text: str) -> str:
    match = re.search(r"<response>(.*?)</response>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"</?response>", "", text, flags=re.IGNORECASE).strip()


def generate_y_pos(client: OpenAIClient, item: ModelInput, level: str, level_notes: str, max_tokens: int) -> str:
    prompt = Y_POS_PROMPT.format(
        question=item.question,
        attempt=item.attempt,
        correct_solution=item.correct_solution,
        error_step=item.error_step,
        error_reason=item.error_reason,
        level=level,
        level_notes=level_notes,
    )
    return strip_response_tags(client.generate(SYSTEM_TUTOR, prompt, max_tokens=max_tokens))


def generate_y_neg(client: OpenAIClient, item: ModelInput, y_pos: str, aspect: str, max_tokens: int) -> str:
    prompt = Y_NEG_PROMPT.format(
        question=item.question,
        attempt=item.attempt,
        correct_solution=item.correct_solution,
        error_step=item.error_step,
        error_reason=item.error_reason,
        good_response=y_pos,
        aspect=aspect,
        aspect_instructions=PEDAGOGICAL_ASPECTS[aspect],
    )
    return strip_response_tags(client.generate(SYSTEM_EDITOR, prompt, max_tokens=max_tokens))


def pair_to_dict(pair: PreferencePair) -> dict[str, Any]:
    return {key: value for key, value in asdict(pair).items() if value is not None}


def pair_key(record: dict[str, Any]) -> tuple[Any, ...]:
    aspects = record.get("targeted_aspects")
    aspects_key: Any = tuple(aspects) if isinstance(aspects, list) else aspects
    return (
        record.get("uuid"),
        record.get("question"),
        record.get("attempt"),
        aspects_key,
        record.get("level"),
    )


def generation_key(item: ModelInput, aspect: str, level: str) -> tuple[Any, ...]:
    return (item.uuid, item.question, item.attempt, (aspect,), level)


def load_existing_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    return {pair_key(record) for record in read_jsonl(path)}


def iter_progress(items: list[ModelInput], description: str) -> Iterable[ModelInput]:
    if tqdm is None:
        return items
    return tqdm(items, desc=description)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def sample_items(
    items: list[ModelInput],
    limit: Optional[int],
    random_sample: bool,
    seed: int,
    start_index: int,
) -> list[ModelInput]:
    sliced = items[start_index:]
    if limit is None:
        return sliced
    limit = min(limit, len(sliced))
    if random_sample:
        rng = random.Random(seed)
        return rng.sample(sliced, k=limit)
    return sliced[:limit]


def prepare_inputs(args: argparse.Namespace) -> list[ModelInput]:
    skip_question_types = set(args.skip_question_types)
    records = []
    for record in iter_dataset_records(args):
        if record.get("question_type", "") in skip_question_types:
            continue
        if args.only_wrong_answers and record.get("model_output_answer_correctness") != "wrong":
            continue
        item = to_model_input(record)
        if item is not None:
            records.append(item)
    return sample_items(
        records,
        limit=args.limit,
        random_sample=args.random_sample,
        seed=args.seed,
        start_index=args.start_index,
    )


def validate_requested_values(args: argparse.Namespace) -> None:
    unknown_aspects = set(args.aspects) - set(PEDAGOGICAL_ASPECTS)
    if unknown_aspects:
        raise SystemExit(f"Unknown aspects: {sorted(unknown_aspects)}")

    unknown_levels = set(args.scaffold_levels) - set(SCAFFOLD_LEVELS)
    if unknown_levels:
        raise SystemExit(f"Unknown scaffold levels: {sorted(unknown_levels)}")

    if args.input and not args.input.exists():
        raise SystemExit(f"Input file does not exist: {args.input}")


def run_generation(args: argparse.Namespace) -> None:
    validate_requested_values(args)
    items = prepare_inputs(args)
    print(f"Prepared {len(items)} MR-GSM8K examples.")

    if args.dry_run:
        for item in items[: min(3, len(items))]:
            print(json.dumps(asdict(item), ensure_ascii=False, indent=2))
        return

    if not os.getenv("OPENAI_API_KEY") and not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key before generation.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        args.output.write_text("", encoding="utf-8")

    seen = load_existing_keys(args.output) if args.resume else set()
    client_pos = OpenAIClient(
        model=args.model_pos,
        temperature=None if args.model_pos.startswith("gpt-5") else args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    client_neg = OpenAIClient(
        model=args.model_neg,
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    generated = 0
    for item in iter_progress(items, "Generating MR-GSM8K pairs"):
        for level in args.scaffold_levels:
            level_notes = SCAFFOLD_LEVELS[level]
            y_pos = generate_y_pos(client_pos, item, level=level, level_notes=level_notes, max_tokens=args.max_tokens_y_pos)

            for aspect in args.aspects:
                pending_key = generation_key(item, aspect, level)
                if pending_key in seen:
                    continue
                y_neg = generate_y_neg(client_neg, item, y_pos=y_pos, aspect=aspect, max_tokens=args.max_tokens_y_neg)
                pair = PreferencePair(
                    uuid=item.uuid,
                    question=item.question,
                    attempt=item.attempt,
                    correct_solution=item.correct_solution,
                    error_step=item.error_step,
                    error_reason=item.error_reason,
                    y_pos=y_pos,
                    y_neg=y_neg,
                    targeted_aspects=[aspect],
                    level=level,
                    level_notes=level_notes,
                    question_type=item.question_type,
                )
                write_jsonl_record(args.output, pair_to_dict(pair))
                seen.add(pending_key)
                generated += 1
                if generated % args.log_every == 0:
                    print(f"Generated {generated} new pairs.")

    print(f"Wrote {generated} new preference pairs to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Optional local MR-GSM8K JSONL. Defaults to Hugging Face.")
    parser.add_argument("--dataset-name", default="Randolphzeng/Mr-GSM8K")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None, help="Optional Hugging Face split. Defaults to all available splits.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-pos", default="gpt-5")
    parser.add_argument("--model-neg", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens-y-pos", type=int, default=5120)
    parser.add_argument("--max-tokens-y-neg", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--skip-question-types", type=parse_csv, default=["POT"])
    parser.add_argument(
        "--only-wrong-answers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only records with model_output_answer_correctness == 'wrong'.",
    )
    parser.add_argument(
        "--aspects",
        type=parse_csv,
        default=list(PEDAGOGICAL_ASPECTS),
        help="Comma-separated degradation aspects. Defaults to all paper aspects.",
    )
    parser.add_argument(
        "--scaffold-levels",
        type=parse_csv,
        default=["L0"],
        help="Comma-separated scaffold levels. Defaults to L0.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_generation(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
