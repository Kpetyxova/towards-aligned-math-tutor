"""Generate RMBoost-style synthetic preference pairs for PRM800K.

This script:
1. filters PRM800K `math_splits` to the paper subset:
   - algebra levels 1-2
   - number theory level 1
   - prealgebra levels 1-3
2. filters PRM800K phase files to those problems and attaches the gold
   solution/metadata from the corresponding `math_splits` file;
3. keeps only examples whose generated solution is marked incorrect
   (`finish_reason == "found_error"`);
4. converts each example to a single dialog turn and generates preference
   pairs: GPT-5 creates the preferred tutor response and GPT-4.1 creates a
   degraded response along one pedagogical aspect.

Example:
    python generate_prm800k_synthetic_pairs.py \
        --prm-root prm800k/prm800k \
        --splits all \
        --phases all \
        --output data/prm800k_synthetic_pairs.jsonl

Set `OPENAI_API_KEY` before running generation. Use `--dry-run` to inspect the
filtered inputs without calling the API.
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
except Exception:
    OpenAI = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


ALLOWED_SUBJECT_LEVELS = {
    ("algebra", 1),
    ("algebra", 2),
    ("number theory", 1),
    ("prealgebra", 1),
    ("prealgebra", 2),
    ("prealgebra", 3),
}

PEDAGOGICAL_ASPECTS = {
    "Factuality": """Factuality + Non-contradiction + No Nonsense: the response should be factually correct, should not contradict what the student said, and should not contain irrelevant information. To degrade this aspect, introduce a clear factual/math error or contradict the student's work while keeping the edit minimal.""",
    "MistakeIdentification": """Mistake Identification: the response should identify, explicitly or implicitly, that there is a mistake in the student's solution. To degrade this aspect, remove or soften the signal that the student's solution is wrong, for example by saying only "nice try" or by validating the work.""",
    "Targetedness": """Targetedness: the response should address the core misconception or misunderstanding in the student's solution. To degrade this aspect, make the feedback generic, focus on an irrelevant part of the solution, or miss the central error.""",
    "RevealingAnswer": """Not revealing the final answer: the tutor may give a substep when needed, but should avoid giving away the final answer. To degrade this aspect, reveal the final answer or enough of the final computation that the student no longer has meaningful work to do.""",
    "Clarity": """Clarity: the tutor's response should be free of awkward, confusing, or misleading wording. To degrade this aspect, keep the same broad idea but make the wording vague, confusing, poorly connected to the student's work, or hard to act on.""",
}

SCAFFOLD_LEVELS = {
    "L0": "Meta-cognitive - tone-first, Socratic nudge, one check question. <=2 sentences.",
}

SYSTEM_TUTOR = "You are an expert math tutor who gives concise, actionable feedback."
SYSTEM_EDITOR = "You are revising feedback to control its quality along requested aspects."

Y1_TUTOR_PROMPT = """
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

Single-turn dialog:
Tutor: Let's work on this problem:
{question}

Student:
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

Y2_WORSE_PROMPT = """
You are an expert tutor teaching another tutor how not to respond to a student's
incorrect math solution.

Only degrade the following aspect:
{aspect}

Aspect-specific degradation instructions:
{aspect_instructions}

Below is the same single-turn dialog:
Tutor: Let's work on this problem:
{question}

Student:
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
    question: str
    attempt: str
    correct_solution: str
    ground_truth_answer: Optional[str]
    error_step: str
    error_reason: str
    prm_split: Optional[str] = None
    prm_phase: Optional[str] = None
    subject: Optional[str] = None
    difficulty_level: Optional[int] = None
    unique_id: Optional[str] = None

    @property
    def dialog(self) -> list[dict[str, str]]:
        return [
            {
                "role": "tutor",
                "content": f"Let's work on this problem:\n{self.question}",
            },
            {"role": "student", "content": self.attempt},
        ]


@dataclass
class PreferencePair:
    question: str
    dialog: list[dict[str, str]]
    attempt: str
    correct_solution: str
    ground_truth_answer: Optional[str]
    error_step: str
    error_reason: str
    y_pos: str
    y_neg: str
    targeted_aspects: list[str]
    scaffold_level: str
    scaffold_level_notes: str
    source: str = "prm800k"
    prm_split: Optional[str] = None
    prm_phase: Optional[str] = None
    subject: Optional[str] = None
    difficulty_level: Optional[int] = None
    unique_id: Optional[str] = None


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
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.model.startswith("gpt-5"):
            payload["reasoning"] = {"effort": "low"}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.responses.create(**payload)
                text = getattr(response, "output_text", "") or ""
                return text.strip()
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(1.5 * attempt)
        raise LLMError(f"OpenAI request failed after retries: {last_error}") from last_error


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_problem_text(text: Any) -> str:
    normalized = str(text).replace("\xa0", " ").replace("\r", "").strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_subject(subject: Any) -> str:
    return " ".join(str(subject).strip().lower().split())


def is_allowed_math_split_record(record: dict[str, Any]) -> bool:
    subject = normalize_subject(record.get("subject", ""))
    try:
        level = int(record.get("level"))
    except Exception:
        return False
    return (subject, level) in ALLOWED_SUBJECT_LEVELS


def load_allowed_math_split_map(math_split_path: Path) -> dict[str, dict[str, Any]]:
    """Map normalized problem text to solution and metadata for allowed records."""
    mapping: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(math_split_path):
        if not is_allowed_math_split_record(record):
            continue
        problem = record.get("problem")
        solution = record.get("solution")
        if not isinstance(problem, str) or not isinstance(solution, str):
            continue
        mapping[normalize_problem_text(problem)] = {
            "solution": solution,
            "answer": record.get("answer"),
            "subject": record.get("subject"),
            "level": record.get("level"),
            "unique_id": record.get("unique_id"),
        }
    return mapping


def filter_phase_records(phase_path: Path, problem_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in read_jsonl(phase_path):
        question = record.get("question") if isinstance(record.get("question"), dict) else {}
        problem = question.get("problem")
        if not isinstance(problem, str):
            continue
        metadata = problem_map.get(normalize_problem_text(problem))
        if metadata is None:
            continue
        question["solution"] = metadata.get("solution")
        question["ground_truth_solution"] = question.get("ground_truth_solution") or metadata.get("solution")
        question["ground_truth_answer"] = question.get("ground_truth_answer") or metadata.get("answer")
        question["subject"] = metadata.get("subject")
        question["level"] = metadata.get("level")
        question["unique_id"] = metadata.get("unique_id")
        record["question"] = question
        filtered.append(record)
    return filtered


def coerce_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item is not None and str(item).strip()]
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if value is not None and str(value).strip() else []


def first_error_step(record: dict[str, Any], generated_steps: list[str]) -> str:
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    for step in label.get("steps", []) or []:
        completions = step.get("completions") or []
        chosen = step.get("chosen_completion")
        if isinstance(chosen, int) and 0 <= chosen < len(completions):
            chosen_completion = completions[chosen]
            if chosen_completion.get("rating") == -1:
                return str(chosen_completion.get("text", "")).strip()
        if chosen is None and step.get("human_completion") is None:
            for completion in completions:
                text = str(completion.get("text", "")).strip()
                if completion.get("rating") == -1 and text in generated_steps:
                    return text
    return ""


def to_model_input(
    record: dict[str, Any],
    *,
    prm_split: Optional[str] = None,
    prm_phase: Optional[str] = None,
) -> Optional[ModelInput]:
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    if label.get("finish_reason") != "found_error":
        return None

    question_obj = record.get("question") if isinstance(record.get("question"), dict) else {}
    question = str(question_obj.get("problem", "")).strip()
    generated_steps = coerce_steps(question_obj.get("pre_generated_steps", []))
    attempt = "\n".join(generated_steps).strip()
    correct_solution = (
        question_obj.get("ground_truth_solution")
        or question_obj.get("solution")
        or record.get("ground_truth_solution")
        or record.get("solution")
        or ""
    )
    ground_truth_answer = (
        question_obj.get("ground_truth_answer")
        or question_obj.get("answer")
        or record.get("ground_truth_answer")
        or record.get("answer")
    )
    if not question or not attempt or not correct_solution:
        return None

    level_value = question_obj.get("level")
    try:
        difficulty_level = int(level_value) if level_value is not None else None
    except Exception:
        difficulty_level = None

    return ModelInput(
        question=question,
        attempt=attempt,
        correct_solution=str(correct_solution).strip(),
        ground_truth_answer=str(ground_truth_answer).strip() if ground_truth_answer else None,
        error_step=first_error_step(record, generated_steps) or "Not available",
        error_reason="Not available",
        prm_split=prm_split,
        prm_phase=prm_phase,
        subject=str(question_obj.get("subject")).strip() if question_obj.get("subject") else None,
        difficulty_level=difficulty_level,
        unique_id=str(question_obj.get("unique_id")).strip() if question_obj.get("unique_id") else None,
    )


def strip_response_tags(text: str) -> str:
    match = re.search(r"<response>(.*?)</response>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"</?response>", "", text, flags=re.IGNORECASE).strip()


def generate_y_pos(
    client: OpenAIClient,
    item: ModelInput,
    scaffold_level: str,
    scaffold_notes: str,
    max_tokens: int,
) -> str:
    prompt = Y1_TUTOR_PROMPT.format(
        question=item.question,
        attempt=item.attempt,
        correct_solution=item.correct_solution,
        error_step=item.error_step,
        error_reason=item.error_reason,
        level=scaffold_level,
        level_notes=scaffold_notes,
    )
    return strip_response_tags(client.generate(SYSTEM_TUTOR, prompt, max_tokens=max_tokens))


def generate_y_neg(
    client: OpenAIClient,
    item: ModelInput,
    y_pos: str,
    aspect: str,
    max_tokens: int,
) -> str:
    prompt = Y2_WORSE_PROMPT.format(
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
    if isinstance(aspects, list):
        aspects_key: Any = tuple(aspects)
    else:
        aspects_key = aspects
    return (
        record.get("question"),
        record.get("attempt"),
        aspects_key,
        record.get("scaffold_level") or record.get("level"),
    )


def generation_key(item: ModelInput, aspect: str, scaffold_level: str) -> tuple[Any, ...]:
    return (item.question, item.attempt, (aspect,), scaffold_level)


def input_key(item: ModelInput) -> tuple[str, str]:
    return (item.question, item.attempt)


def dedupe_inputs(inputs: list[ModelInput]) -> list[ModelInput]:
    seen: set[tuple[str, str]] = set()
    unique_inputs: list[ModelInput] = []
    for item in inputs:
        key = input_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique_inputs.append(item)
    return unique_inputs


def load_existing_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[Any, ...]] = set()
    for record in read_jsonl(path):
        keys.add(pair_key(record))
    return keys


def iter_progress(items: list[ModelInput], description: str) -> Iterable[ModelInput]:
    if tqdm is None:
        return items
    return tqdm(items, desc=description)


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


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def split_sort_key(value: str) -> tuple[int, list[Any]]:
    preferred_order = {"train": 0, "test": 1}
    return (preferred_order.get(value, 2), natural_sort_key(value))


def discover_splits(prm_root: Path) -> list[str]:
    math_splits_dir = prm_root / "math_splits"
    phase_data_dir = prm_root / "data"
    math_split_names = {path.stem for path in math_splits_dir.glob("*.jsonl") if "." not in path.stem}
    phase_split_names: set[str] = set()
    for path in phase_data_dir.glob("phase*_*.jsonl"):
        match = re.fullmatch(r"(phase[^_]+)_(.+)\.jsonl", path.name)
        if match:
            phase_split_names.add(match.group(2))
    return sorted(math_split_names & phase_split_names, key=split_sort_key)


def discover_phases(prm_root: Path, splits: list[str]) -> list[str]:
    phase_data_dir = prm_root / "data"
    split_set = set(splits)
    phases: set[str] = set()
    for path in phase_data_dir.glob("phase*_*.jsonl"):
        match = re.fullmatch(r"(phase[^_]+)_(.+)\.jsonl", path.name)
        if match and match.group(2) in split_set:
            phases.add(match.group(1))
    return sorted(phases, key=natural_sort_key)


def resolve_requested_splits(args: argparse.Namespace) -> list[str]:
    if args.splits and args.split:
        raise SystemExit("Use either --splits or --split, not both.")
    requested = args.splits or ([args.split] if args.split else ["all"])
    if "all" in requested:
        splits = discover_splits(args.prm_root)
    else:
        splits = requested
    if not splits:
        raise SystemExit(f"No PRM800K splits found under {args.prm_root / 'math_splits'}")
    return splits


def resolve_requested_phases(args: argparse.Namespace, splits: list[str]) -> list[str]:
    if args.phases and args.phase:
        raise SystemExit("Use either --phases or --phase, not both.")
    requested = args.phases or ([args.phase] if args.phase else ["all"])
    if "all" in requested:
        phases = discover_phases(args.prm_root, splits)
    else:
        phases = requested
    if not phases:
        raise SystemExit(f"No PRM800K phases found under {args.prm_root / 'data'}")
    return phases


def prepare_inputs(args: argparse.Namespace) -> list[ModelInput]:
    if args.phase_input:
        phase_records = list(read_jsonl(args.phase_input))
        inputs = [item for record in phase_records if (item := to_model_input(record)) is not None]
    else:
        inputs = []
        filtered_phase_records: list[dict[str, Any]] = []
        splits = resolve_requested_splits(args)
        phases = resolve_requested_phases(args, splits)
        for split in splits:
            math_split_path = args.prm_root / "math_splits" / f"{split}.jsonl"
            if not math_split_path.exists():
                raise SystemExit(f"Math split does not exist: {math_split_path}")
            problem_map = load_allowed_math_split_map(math_split_path)
            for phase in phases:
                phase_path = args.prm_root / "data" / f"{phase}_{split}.jsonl"
                if not phase_path.exists():
                    raise SystemExit(f"PRM phase file does not exist: {phase_path}")
                phase_records = filter_phase_records(phase_path, problem_map)
                filtered_phase_records.extend(phase_records)
                inputs.extend(
                    item
                    for record in phase_records
                    if (item := to_model_input(record, prm_split=split, prm_phase=phase)) is not None
                )

        if args.write_filtered_phase:
            args.write_filtered_phase.parent.mkdir(parents=True, exist_ok=True)
            with args.write_filtered_phase.open("w", encoding="utf-8") as fh:
                for record in filtered_phase_records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    inputs = dedupe_inputs(inputs)
    return sample_items(
        inputs,
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

    requested_splits = args.splits or ([args.split] if args.split else [])
    if "all" in requested_splits and len(requested_splits) > 1:
        raise SystemExit("Use --splits all by itself, or list concrete splits.")

    requested_phases = args.phases or ([args.phase] if args.phase else [])
    if "all" in requested_phases and len(requested_phases) > 1:
        raise SystemExit("Use --phases all by itself, or list concrete phases.")

    if not args.phase_input and not args.prm_root.exists():
        raise SystemExit(f"PRM root does not exist: {args.prm_root}")


def run_generation(args: argparse.Namespace) -> None:
    validate_requested_values(args)
    items = prepare_inputs(args)
    print(f"Prepared {len(items)} incorrect PRM800K examples.")

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
        temperature=args.temperature,
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
    for item in iter_progress(items, "Generating PRM800K pairs"):
        for scaffold_level in args.scaffold_levels:
            scaffold_notes = SCAFFOLD_LEVELS[scaffold_level]
            y_pos = generate_y_pos(
                client_pos,
                item,
                scaffold_level=scaffold_level,
                scaffold_notes=scaffold_notes,
                max_tokens=args.max_tokens_y_pos,
            )

            for aspect in args.aspects:
                pending_key = generation_key(item, aspect, scaffold_level)
                if pending_key in seen:
                    continue
                y_neg = generate_y_neg(
                    client_neg,
                    item,
                    y_pos=y_pos,
                    aspect=aspect,
                    max_tokens=args.max_tokens_y_neg,
                )
                pair = PreferencePair(
                    question=item.question,
                    dialog=item.dialog,
                    attempt=item.attempt,
                    correct_solution=item.correct_solution,
                    ground_truth_answer=item.ground_truth_answer,
                    error_step=item.error_step,
                    error_reason=item.error_reason,
                    y_pos=y_pos,
                    y_neg=y_neg,
                    targeted_aspects=[aspect],
                    scaffold_level=scaffold_level,
                    scaffold_level_notes=scaffold_notes,
                    prm_split=item.prm_split,
                    prm_phase=item.prm_phase,
                    subject=item.subject,
                    difficulty_level=item.difficulty_level,
                    unique_id=item.unique_id,
                )
                record = pair_to_dict(pair)
                write_jsonl_record(args.output, record)
                seen.add(generation_key(item, aspect, scaffold_level))
                generated += 1
                if generated % args.log_every == 0:
                    print(f"Generated {generated} new pairs.")

    print(f"Wrote {generated} new preference pairs to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prm-root", type=Path, default=Path("prm800k/prm800k"))
    parser.add_argument(
        "--splits",
        type=parse_csv,
        default=None,
        help="Comma-separated PRM800K splits, or 'all'. Defaults to all discovered splits.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Single split alias for --splits. Use 'train', 'test', or 'all'.",
    )
    parser.add_argument(
        "--phases",
        type=parse_csv,
        default=None,
        help="Comma-separated PRM800K phases, or 'all'. Defaults to all discovered phases.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Single phase alias for --phases. Use values like 'phase1', 'phase2', or 'all'.",
    )
    parser.add_argument(
        "--phase-input",
        type=Path,
        default=None,
        help="Optional pre-filtered PRM800K phase JSONL. If provided, --prm-root filtering is skipped.",
    )
    parser.add_argument(
        "--write-filtered-phase",
        type=Path,
        default=None,
        help="Optional path for the filtered phase JSONL before LLM generation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-pos", default="gpt-5")
    parser.add_argument("--model-neg", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens-y-pos", type=int, default=512)
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
