"""Prepare train/dev/test files from generated tutor-response datasets.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


SPLITS = ("train", "dev", "test")
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


def resolve_path(path: Path, *, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or JSONL file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected object records in {path} at line {line_number}")
        records.append(record)
    return records


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


def extract_final_answer_from_gsm8k(raw_answer: Any) -> str:
    text = "" if raw_answer is None else str(raw_answer)
    if "####" in text:
        return text.split("####", 1)[1].strip()
    return text.strip()


def canonicalize_number_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
    except Exception:
        return text
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return text
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.15g}"


def parse_mawps_numbers(numbers_field: Any) -> list[str]:
    if numbers_field is None:
        return []
    if isinstance(numbers_field, str):
        return [canonicalize_number_text(part) for part in numbers_field.split()]
    if isinstance(numbers_field, (list, tuple)):
        return [canonicalize_number_text(value) for value in numbers_field]
    return []


def replace_mawps_placeholders(text: Any, numbers: list[str]) -> str:
    if text is None:
        return ""
    source = str(text)

    def repl(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 0 <= index < len(numbers):
            return numbers[index]
        return match.group(0)

    return re.sub(r"\bN_(\d{1,2})\b", repl, source)


def normalize_problem_solving_prm(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, record in enumerate(records):
        normalized.append(
            {
                "problem": record.get("problem", ""),
                "solution": record.get("solution", ""),
                "answer": "" if record.get("answer") is None else str(record.get("answer")),
                "dataset": "PRM800K",
                "id": record.get("unique_id") or record.get("id") or f"prm800k_{idx}",
            }
        )
    return normalized


def normalize_problem_solving_gsm8k(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, record in enumerate(records):
        raw_answer = record.get("answer") or record.get("final_answer") or record.get("ans") or ""
        normalized.append(
            {
                "problem": record.get("question") or record.get("problem") or record.get("prompt") or "",
                "solution": record.get("solution") or record.get("rationale") or record.get("cot") or str(raw_answer),
                "answer": extract_final_answer_from_gsm8k(raw_answer),
                "dataset": "GSM8K",
                "id": str(record.get("id") or record.get("idx") or f"gsm8k_{idx}"),
            }
        )
    return normalized


def normalize_problem_solving_mawps(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, record in enumerate(records):
        numbers = parse_mawps_numbers(record.get("Numbers") or record.get("numbers"))
        problem = replace_mawps_placeholders(
            record.get("question") or record.get("Question") or record.get("sQuestion") or record.get("Problem") or "",
            numbers,
        )
        equation = record.get("equation") or record.get("Equation")
        if equation is not None:
            solution = replace_mawps_placeholders(equation, numbers)
        else:
            solution_value = record.get("solution") or record.get("rationale") or record.get("lSolutions") or ""
            if isinstance(solution_value, list):
                solution = "\n".join(str(value) for value in solution_value)
            else:
                solution = str(solution_value)
        answer = record.get("answer") or record.get("Answer") or record.get("ans") or record.get("final_answer") or ""
        normalized.append(
            {
                "problem": problem,
                "solution": solution,
                "answer": canonicalize_number_text(answer),
                "dataset": "MAWPS",
                "id": str(record.get("id") or record.get("iIndex") or record.get("uid") or f"mawps_{idx}"),
            }
        )
    return normalized


def load_hf_records(
    dataset_name: str,
    config_name: str | None = None,
    *,
    split: str | None = None,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    try:
        from datasets import DownloadConfig, load_dataset  # type: ignore
    except Exception:
        return []

    kwargs: dict[str, Any] = {}
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)

    def _load() -> Any:
        if config_name is None:
            return load_dataset(dataset_name, split=split, **kwargs)
        return load_dataset(dataset_name, config_name, split=split, **kwargs)

    try:
        loaded = _load()
    except TypeError:
        kwargs.pop("download_config", None)
        try:
            loaded = _load()
        except Exception:
            return []
    except Exception:
        return []

    records: list[dict[str, Any]] = []
    if hasattr(loaded, "items"):
        datasets = loaded.values()
    else:
        datasets = [loaded]
    for dataset in datasets:
        for row in dataset:
            records.append(dict(row))
    return records


def sample_problem_solving_sources(
    *,
    prm_math_splits_dir: Path,
    sample_fraction: float,
    seed: int,
    local_hf_only: bool,
) -> list[dict[str, Any]]:
    prm_records: list[dict[str, Any]] = []
    for filename in ("train.filtered.algebra_nt_prealgebra.jsonl", "test.filtered.algebra_nt_prealgebra.jsonl"):
        path = prm_math_splits_dir / filename
        if path.exists():
            prm_records.extend(load_records(path))
    prm_all = normalize_problem_solving_prm(prm_records)

    gsm8k_all = normalize_problem_solving_gsm8k(
        load_hf_records("openai/gsm8k", "main", local_files_only=local_hf_only)
    )
    mawps_raw = load_hf_records("mwpt5/MAWPS", split="train", local_files_only=local_hf_only)
    if not mawps_raw:
        mawps_raw = load_hf_records("mwpt5/MAWPS", local_files_only=local_hf_only)
    mawps_all = normalize_problem_solving_mawps(mawps_raw)

    counts = {"PRM800K": len(prm_all), "GSM8K": len(gsm8k_all), "MAWPS": len(mawps_all)}
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        hint = "pass --allow-hf-downloads if these Hugging Face datasets are not already cached"
        raise RuntimeError(f"Could not build problem-solving sample; missing {missing}. {hint}.")

    rng = random.Random(seed)

    def sample_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        k = max(1, int(round(len(items) * sample_fraction)))
        return rng.sample(items, min(k, len(items)))

    return sample_list(prm_all) + sample_list(gsm8k_all) + sample_list(mawps_all)


def ensure_problem_solving_sample(
    path: Path,
    *,
    prm_math_splits_dir: Path,
    sample_fraction: float,
    seed: int,
    rebuild: bool,
    local_hf_only: bool,
) -> None:
    if path.exists() and not rebuild:
        return
    sample = sample_problem_solving_sources(
        prm_math_splits_dir=prm_math_splits_dir,
        sample_fraction=sample_fraction,
        seed=seed,
        local_hf_only=local_hf_only,
    )
    write_json(path, sample)


def split_80_10_10(records: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    dev_end = int(len(shuffled) * 0.9)
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


def split_with_test_overlap_to_train(
    records: list[dict[str, Any]],
    *,
    question_getter,
    test_problems: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    filtered = [item for item in records if question_getter(item) not in test_problems]
    overlap = [item for item in records if question_getter(item) in test_problems]
    train, dev, test = split_80_10_10(filtered, seed=seed)
    return overlap + train, dev, test


def dialog_history(question: str, attempt: str) -> str:
    return f"Tutor: {question}\nStudent: {attempt}\n"


def format_dialog_item(
    item: dict[str, Any],
    *,
    dataset: str,
    item_id: Any,
    question: str,
    attempt: str,
    tutor_response: str,
    gold_solution: str,
    is_correct: bool,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "dataset": dataset,
        "dialog_history": dialog_history(question, attempt),
        "tutor_response": tutor_response,
        "gold_solution": gold_solution,
        "is_correct": is_correct,
    }


def format_preference_dialog(item: dict[str, Any], *, dataset: str, item_id: Any) -> dict[str, Any]:
    return format_dialog_item(
        item,
        dataset=dataset,
        item_id=item_id,
        question=str(item.get("question", "")),
        attempt=str(item.get("attempt", "")),
        tutor_response=str(item.get("y_pos", "")),
        gold_solution=str(item.get("correct_solution", "")),
        is_correct=False,
    )


def format_dpo_item(item: dict[str, Any], *, dataset: str, item_id: Any) -> dict[str, Any]:
    return {
        "id": item_id,
        "dataset": dataset,
        "dialog_history": dialog_history(str(item.get("question", "")), str(item.get("attempt", ""))),
        "preferred": item.get("y_pos", ""),
        "non_preferred": item.get("y_neg", ""),
        "gold_solution": item.get("correct_solution", ""),
        "aspect": (item.get("targeted_aspects") or [""])[0],
        "is_correct": False,
    }


def split_preference_source(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    test_problems: set[str],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Split generated preference rows into SFT and DPO files.

    The notebook uses only `Factuality` rows to form SFT feedback splits, then
    includes all aspects in DPO according to those same question/attempt splits.
    """
    l0_rows = [row for row in rows if (row.get("level") or row.get("scaffold_level")) == "L0"]
    factuality_rows = [row for row in l0_rows if row.get("targeted_aspects") == ["Factuality"]]
    train_base, dev_base, test_base = split_with_test_overlap_to_train(
        factuality_rows,
        question_getter=lambda item: str(item.get("question", "")),
        test_problems=test_problems,
        seed=seed,
    )

    base_by_split = {"train": train_base, "dev": dev_base, "test": test_base}
    dialog_splits: dict[str, list[dict[str, Any]]] = {}
    dpo_splits: dict[str, list[dict[str, Any]]] = {}
    key_sets = {
        split: {(item.get("question"), item.get("attempt")) for item in split_rows}
        for split, split_rows in base_by_split.items()
    }

    for split, split_rows in base_by_split.items():
        dialog_splits[split] = [
            format_preference_dialog(item, dataset=dataset_name, item_id=item.get("uuid") or idx)
            for idx, item in enumerate(split_rows)
        ]

        dpo_rows = [
            item
            for item in l0_rows
            if (item.get("question"), item.get("attempt")) in key_sets[split]
        ]
        dpo_splits[split] = [
            format_dpo_item(item, dataset=dataset_name, item_id=item.get("uuid") or idx)
            for idx, item in enumerate(dpo_rows)
        ]

    return dialog_splits, dpo_splits


def extract_prm_correct_question(item: dict[str, Any]) -> str:
    question = item.get("question")
    if isinstance(question, dict):
        return str(question.get("problem", ""))
    return str(item.get("question", ""))


def extract_prm_correct_solution(item: dict[str, Any]) -> str:
    question = item.get("question") if isinstance(item.get("question"), dict) else {}
    return str(question.get("solution") or question.get("ground_truth_solution") or item.get("correct_solution") or "")


def extract_prm_correct_attempt(item: dict[str, Any]) -> str:
    question = item.get("question") if isinstance(item.get("question"), dict) else {}
    pre_generated_steps = question.get("pre_generated_steps")
    if isinstance(pre_generated_steps, list) and pre_generated_steps:
        return " ".join(str(step) for step in pre_generated_steps)

    label = item.get("label") if isinstance(item.get("label"), dict) else {}
    steps = label.get("steps") or []
    attempt_parts: list[str] = []
    for step in steps:
        completions = step.get("completions") or []
        chosen = step.get("chosen_completion")
        if isinstance(chosen, int) and 0 <= chosen < len(completions):
            attempt_parts.append(str(completions[chosen].get("text", "")))
        else:
            human = step.get("human_completion") or {}
            if isinstance(human, dict):
                attempt_parts.append(str(human.get("text", "")))
    return " ".join(part for part in attempt_parts if part)


def split_correct_source(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    question_getter,
    attempt_getter,
    solution_getter,
    id_getter,
    test_problems: set[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    train, dev, test = split_with_test_overlap_to_train(
        rows,
        question_getter=question_getter,
        test_problems=test_problems,
        seed=seed,
    )
    raw_splits = {"train": train, "dev": dev, "test": test}
    formatted: dict[str, list[dict[str, Any]]] = {}
    for split, split_rows in raw_splits.items():
        formatted[split] = [
            format_dialog_item(
                item,
                dataset=dataset_name,
                item_id=id_getter(item, idx),
                question=question_getter(item),
                attempt=attempt_getter(item),
                tutor_response=str(item.get("tutor_response") or item.get("y_pos") or ""),
                gold_solution=solution_getter(item),
                is_correct=True,
            )
            for idx, item in enumerate(split_rows)
        ]
    return formatted


def clean_mathdial_intents(text: str) -> str:
    cleaned = re.sub(r"(?<=:)\s*\([^)]*\)\s*", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned.strip()


def clean_mathdial_tutor_response(turn: str) -> str:
    return re.sub(r"^Teacher:\s*", "", turn).strip()


def format_mathdial_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for item in rows:
        turns = [turn.strip() for turn in clean_mathdial_intents(str(item.get("conversation", ""))).split("|EOM|") if turn.strip()]
        teacher_turn_indices = [idx for idx, turn in enumerate(turns) if turn.startswith("Teacher:")]
        last_teacher_turn_index = teacher_turn_indices[-1] if teacher_turn_indices else None
        self_correctness = str(item.get("self-correctness", "")).strip().lower()
        final_turn_is_correct = self_correctness != "no"
        for turn_index, turn in enumerate(turns):
            if not turn.startswith("Teacher:"):
                continue
            formatted.append(
                {
                    "id": item.get("qid"),
                    "dataset": "mathdial",
                    "dialog_history": (
                        f"Tutor: {item.get('question', '')}\n"
                        f"Student: {item.get('student_incorrect_solution', '')}\n"
                        + "\n".join(turns[:turn_index])
                    ),
                    "tutor_response": clean_mathdial_tutor_response(turn),
                    "gold_solution": item.get("ground_truth", ""),
                    "is_correct": bool(turn_index == last_teacher_turn_index and final_turn_is_correct),
                }
            )
    return formatted


def split_mathdial_source(
    rows: list[dict[str, Any]],
    *,
    test_problems: set[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    train, dev, test = split_with_test_overlap_to_train(
        rows,
        question_getter=lambda item: str(item.get("question", "")),
        test_problems=test_problems,
        seed=seed,
    )
    return {
        "train": format_mathdial_records(train),
        "dev": format_mathdial_records(dev),
        "test": format_mathdial_records(test),
    }


def split_grouped_dialog_instances(
    instances_by_id: dict[str, list[dict[str, Any]]],
    *,
    overlap: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    ids = list(instances_by_id)
    rng.shuffle(ids)
    train_end = int(len(ids) * 0.8)
    dev_end = int(len(ids) * 0.9)
    train_ids = ids[:train_end]
    dev_ids = ids[train_end:dev_end]
    test_ids = ids[dev_end:]
    return {
        "train": overlap + [instance for item_id in train_ids for instance in instances_by_id[item_id]],
        "dev": [instance for item_id in dev_ids for instance in instances_by_id[item_id]],
        "test": [instance for item_id in test_ids for instance in instances_by_id[item_id]],
    }


def split_socra_multi_source(
    data: dict[str, Any],
    *,
    test_problems: set[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    instances_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlap: list[dict[str, Any]] = []
    for value in data.values():
        if not isinstance(value, dict):
            continue
        question = str(value.get("question", ""))
        analysis = str(value.get("analysis", ""))
        dialogues = value.get("dialogues") or {}
        if not isinstance(dialogues, dict):
            continue
        for name, dialog in dialogues.items():
            if not isinstance(dialog, list):
                continue
            system_turn_indices = [
                idx
                for idx, turn in enumerate(dialog)
                if idx != 0 and isinstance(turn, dict) and "system" in turn
            ]
            last_system_turn_index = system_turn_indices[-1] if system_turn_indices else None
            for turn_index, turn in enumerate(dialog):
                if turn_index == 0 or not isinstance(turn, dict) or "system" not in turn:
                    continue
                previous_turns = f"Tutor: {question}\n"
                for previous_turn in dialog[:turn_index]:
                    if not isinstance(previous_turn, dict):
                        continue
                    previous_turns += f"Tutor: {previous_turn.get('system', '')}\n"
                    previous_turns += f"Student: {previous_turn.get('user', '')}\n"
                instance = {
                    "id": name,
                    "dataset": "socra_teach_multi",
                    "dialog_history": previous_turns,
                    "tutor_response": turn.get("system", ""),
                    "gold_solution": analysis,
                    "is_correct": turn_index == last_system_turn_index,
                }
                if question in test_problems:
                    overlap.append(instance)
                else:
                    instances_by_id[str(name)].append(instance)
    return split_grouped_dialog_instances(instances_by_id, overlap=overlap, seed=seed)


def extract_socra_single_problem(text: str) -> str:
    match = re.search(r"\[Problem\](.*?)\s*\[Answer\]", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_socra_single_analysis(text: str) -> str:
    match = re.search(r"\[Analysis\]\s*(.*)$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def format_socra_single_record(key: str, sample: dict[str, Any]) -> dict[str, Any] | None:
    history = sample.get("history") or []
    if not isinstance(history, list) or not history:
        return None
    first_turn = history[0]
    if not isinstance(first_turn, list) or len(first_turn) < 2:
        return None
    prompt_text = str(first_turn[0])
    question = extract_socra_single_problem(prompt_text)
    dialog_history = f"Tutor: {question}\n{first_turn[1]}\n"
    for turn in history[1:]:
        if isinstance(turn, list) and len(turn) >= 2:
            dialog_history += f"Student: {turn[0]}\nTutor: {turn[1]}\n"
    dialog_history += f"Student: {sample.get('prompt', '')}\n"
    return {
        "id": key,
        "dataset": "socra_teach_single",
        "dialog_history": dialog_history,
        "tutor_response": sample.get("response", ""),
        "gold_solution": extract_socra_single_analysis(prompt_text),
        "is_correct": "incorrect" not in key.lower(),
        "_question": question,
    }


def split_socra_single_source(
    data: dict[str, Any],
    *,
    test_problems: set[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    formatted = []
    overlap = []
    for key, sample in data.items():
        if not isinstance(sample, dict):
            continue
        instance = format_socra_single_record(str(key), sample)
        if instance is None:
            continue
        question = str(instance.pop("_question"))
        if question in test_problems:
            overlap.append(instance)
        else:
            formatted.append(instance)
    train, dev, test = split_80_10_10(formatted, seed=seed)
    return {"train": overlap + train, "dev": dev, "test": test}


def write_split_group(base_dir: Path, prefix: str, splits: dict[str, list[dict[str, Any]]]) -> None:
    for split in SPLITS:
        write_json(base_dir / f"{prefix}_{split}.json", splits[split])


def write_dpo_group(base_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    for split in SPLITS:
        write_json(base_dir / f"dpo_{split}.json", splits[split])


def summarize(name: str, splits: dict[str, list[dict[str, Any]]]) -> None:
    counts = {split: len(rows) for split, rows in splits.items()}
    print(f"{name}: {counts}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "training_sets")
    parser.add_argument("--problem-solving", type=Path, default=DATA_DIR / "samples_proportional_10pct.json")
    parser.add_argument("--rebuild-problem-solving", action="store_true")
    parser.add_argument("--allow-hf-downloads", action="store_true")
    parser.add_argument("--problem-solving-sample-fraction", type=float, default=0.10)
    parser.add_argument("--prm-math-splits-dir", type=Path, default=DATA_DIR / "prm800k" / "prm800k" / "math_splits")
    parser.add_argument("--mathdial-train", type=Path, default=DATA_DIR / "mathdial" / "train-2.jsonl")
    parser.add_argument("--mathdial-test", type=Path, default=DATA_DIR / "mathdial" / "test-2.jsonl")
    parser.add_argument("--socra-multi", type=Path, default=DATA_DIR / "SocraticLM" / "data" / "SocraTeach_multi.json")
    parser.add_argument("--socra-single", type=Path, default=DATA_DIR / "SocraticLM" / "data" / "SocraTeach_single.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data_dir = resolve_path(args.data_dir, base=SCRIPT_DIR)
    output_dir = resolve_path(args.output_dir, base=SCRIPT_DIR)
    dialog_dir = output_dir / "dialog_data_ft"
    dpo_dir = output_dir / "dpo"

    problem_solving_path = resolve_path(args.problem_solving, base=SCRIPT_DIR)
    prm_math_splits_dir = resolve_path(args.prm_math_splits_dir, base=SCRIPT_DIR)
    ensure_problem_solving_sample(
        problem_solving_path,
        prm_math_splits_dir=prm_math_splits_dir,
        sample_fraction=args.problem_solving_sample_fraction,
        seed=args.seed,
        rebuild=args.rebuild_problem_solving,
        local_hf_only=not args.allow_hf_downloads,
    )
    problem_solving = load_records(problem_solving_path)
    problem_train, problem_dev, problem_test = split_80_10_10(problem_solving, seed=args.seed)
    test_problems = {str(item.get("problem", "")) for item in problem_test}
    print(f"problem_solving reference split: {{'train': {len(problem_train)}, 'dev': {len(problem_dev)}, 'test': {len(problem_test)}}}")

    prm_pref = load_records(data_dir / "prm800k_synthetic_pairs.jsonl")
    mr_pref = load_records(data_dir / "mr_gsm8k_synthetic_pairs.jsonl")
    prm_correct = load_records(data_dir / "prm800k_correct_solution_responses.jsonl")
    mr_correct = load_records(data_dir / "mr-gsm8k_correct_solution_responses.jsonl")
    mathdial = load_records(resolve_path(args.mathdial_train, base=SCRIPT_DIR)) + load_records(
        resolve_path(args.mathdial_test, base=SCRIPT_DIR)
    )
    socra_multi_raw = load_json(resolve_path(args.socra_multi, base=SCRIPT_DIR))
    socra_single_raw = load_json(resolve_path(args.socra_single, base=SCRIPT_DIR))

    prm_dialog, prm_dpo = split_preference_source(
        prm_pref,
        dataset_name="prm800k_l0",
        test_problems=test_problems,
        seed=args.seed,
    )
    mr_dialog, mr_dpo = split_preference_source(
        mr_pref,
        dataset_name="mr_gsm8k_l0",
        test_problems=test_problems,
        seed=args.seed,
    )

    prm_correct_dialog = split_correct_source(
        prm_correct,
        dataset_name="prm800k_correct",
        question_getter=extract_prm_correct_question,
        attempt_getter=extract_prm_correct_attempt,
        solution_getter=extract_prm_correct_solution,
        id_getter=lambda _item, idx: idx,
        test_problems=test_problems,
        seed=args.seed,
    )
    mr_correct_dialog = split_correct_source(
        mr_correct,
        dataset_name="mr_gsm8k_correct",
        question_getter=lambda item: str(item.get("question", "")),
        attempt_getter=lambda item: str(item.get("attempt", "")),
        solution_getter=lambda item: str(item.get("correct_solution", "")),
        id_getter=lambda item, idx: item.get("uuid") or idx,
        test_problems=test_problems,
        seed=args.seed,
    )
    mathdial_dialog = split_mathdial_source(mathdial, test_problems=test_problems, seed=args.seed)
    socra_multi_dialog = split_socra_multi_source(socra_multi_raw, test_problems=test_problems, seed=args.seed)
    socra_single_dialog = split_socra_single_source(socra_single_raw, test_problems=test_problems, seed=args.seed)

    dialog_base = {
        split: (
            mathdial_dialog[split]
            + mr_correct_dialog[split]
            + prm_correct_dialog[split]
            + socra_multi_dialog[split]
            + socra_single_dialog[split]
        )
        for split in SPLITS
    }
    dialog_l0 = {
        split: mr_dialog[split] + prm_dialog[split] + dialog_base[split]
        for split in SPLITS
    }
    dpo_l0 = {split: mr_dpo[split] + prm_dpo[split] for split in SPLITS}

    write_split_group(dialog_dir, "dialog_data_ft", dialog_l0)
    write_dpo_group(dpo_dir, dpo_l0)
    summarize("dialog_data_ft", dialog_l0)
    summarize("dpo", dpo_l0)

    for split in SPLITS:
        print(f"{split} dialog datasets: {dict(Counter(item['dataset'] for item in dialog_l0[split]))}")
        print(f"{split} dpo datasets: {dict(Counter(item['dataset'] for item in dpo_l0[split]))}")


if __name__ == "__main__":
    main()
