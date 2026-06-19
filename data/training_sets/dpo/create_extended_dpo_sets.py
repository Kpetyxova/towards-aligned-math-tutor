#!/usr/bin/env python3
"""Create DPO `_extended` files with synthetic number-factuality pairs.

This is a script version of `create_preference_pairs_with_numbers_changed.ipynb`.
It samples tutor responses containing numbers from the combined feedback splits,
perturbs one number or unit in each sampled response, then appends those synthetic
preference pairs to the existing DPO split.

Default inputs, relative to this script:

- `../dialog_data_ft/dialog_data_ft_train.json`
- `../dialog_data_ft/dialog_data_ft_dev.json`
- `../dialog_data_ft/dialog_data_ft_test.json`
- `dpo_train.json`
- `dpo_dev.json`
- `dpo_test.json`

Default outputs:

- `dpo_train_extended.json`
- `dpo_dev_extended.json`
- `dpo_test_extended.json`
"""

from __future__ import annotations

import argparse
import json
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "dev", "test")
DEFAULT_SAMPLE_SIZES = {"train": 20_000, "dev": 2_000, "test": 2_000}
DEFAULT_SEEDS = {"train": 42, "dev": 43, "test": 44}

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SETS_DIR = SCRIPT_DIR.parent
DEFAULT_DIALOG_DIR = TRAINING_SETS_DIR / "dialog_data_ft"
DEFAULT_DPO_DIR = SCRIPT_DIR

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
NUMBER_WITH_UNIT_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)(\s*)([A-Za-z][A-Za-z/%]*)\b")

UNIT_MAP = {
    "hectare": "acre",
    "hectares": "acres",
    "acre": "hectare",
    "acres": "hectares",
    "meter": "foot",
    "meters": "feet",
    "metre": "foot",
    "metres": "feet",
    "km": "miles",
    "kms": "miles",
    "kilometer": "mile",
    "kilometers": "miles",
    "kilometre": "mile",
    "kilometres": "miles",
    "cm": "inch",
    "cms": "inches",
    "centimeter": "inch",
    "centimeters": "inches",
    "g": "oz",
    "kg": "lb",
    "kgs": "lbs",
    "gram": "ounce",
    "grams": "ounces",
    "kilogram": "pound",
    "kilograms": "pounds",
    "l": "gal",
    "liter": "gallon",
    "liters": "gallons",
    "litre": "gallon",
    "litres": "gallons",
}


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def stream_json_array(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except Exception:
        yield from load_json_array(path)
        return

    with path.open("r", encoding="utf-8") as fh:
        for row in ijson.items(fh, "item"):
            if isinstance(row, dict):
                yield row


def format_decimal(value: float, template: str) -> str:
    if "." in template:
        decimals = len(template.split(".", 1)[1])
        return f"{value:.{decimals}f}"
    return f"{int(value)}"


def mutate_number(num_str: str, strategy: str, rng: random.Random) -> str | None:
    if strategy == "digit_swap":
        if "/" in num_str or "." in num_str or len(num_str) < 2:
            return None
        i, j = rng.sample(range(len(num_str)), 2)
        chars = list(num_str)
        chars[i], chars[j] = chars[j], chars[i]
        mutated = "".join(chars)
        return None if mutated == num_str else mutated

    if strategy == "fraction_change":
        if "/" not in num_str:
            return None
        try:
            numerator, denominator = num_str.split("/", 1)
            numerator_i, denominator_i = int(numerator), int(denominator)
        except ValueError:
            return None
        if rng.random() < 0.5:
            numerator_i = max(1, numerator_i + rng.choice([-1, 1]))
        else:
            denominator_i = max(1, denominator_i + rng.choice([-1, 1]))
        mutated = f"{numerator_i}/{denominator_i}"
        return None if mutated == num_str else mutated

    if strategy in {"off_by_one", "off_by_small"}:
        delta = rng.choice([-1, 1]) if strategy == "off_by_one" else rng.randint(2, 5)
        try:
            if "/" in num_str:
                return str(Fraction(num_str) + delta)
            if "." in num_str:
                mutated = format_decimal(float(num_str) + delta, num_str)
                return mutated if mutated != num_str else None
            mutated = str(int(num_str) + delta)
            return mutated if mutated != num_str else None
        except Exception:
            return None

    return None


def apply_unit_error(text: str, rng: random.Random) -> str | None:
    candidates = []
    for match in NUMBER_WITH_UNIT_PATTERN.finditer(text):
        unit = match.group(3)
        replacement = UNIT_MAP.get(unit.lower())
        if replacement:
            candidates.append((match.span(3), replacement))
    if not candidates:
        return None

    (start, end), replacement = rng.choice(candidates)
    mutated = text[:start] + replacement + text[end:]
    return None if mutated == text else mutated


def perturb_response(response: str, rng: random.Random) -> str | None:
    number_matches = list(NUMBER_PATTERN.finditer(response))
    if not number_matches:
        return None

    strategies = ["unit_error", "off_by_one", "off_by_small", "digit_swap", "fraction_change"]
    rng.shuffle(strategies)

    for strategy in strategies:
        if strategy == "unit_error":
            mutated = apply_unit_error(response, rng)
            if mutated and mutated != response:
                return mutated
            continue

        match = rng.choice(number_matches)
        mutated_number = mutate_number(match.group(), strategy, rng)
        if not mutated_number or mutated_number == match.group():
            continue

        start, end = match.span()
        mutated = response[:start] + mutated_number + response[end:]
        if mutated != response:
            return mutated

    return None


def sample_pairs_with_number_noise(
    source_path: Path,
    *,
    sample_size: int,
    seed: int,
    allow_smaller: bool,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    eligible = 0

    for row in stream_json_array(source_path):
        tutor_response = row.get("tutor_response", "")
        if not tutor_response or not re.search(r"\d", str(tutor_response)):
            continue

        mutated = perturb_response(str(tutor_response), rng)
        if not mutated:
            continue

        eligible += 1
        sample = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "dialog_history": row.get("dialog_history", ""),
            "preferred": tutor_response,
            "non_preferred": mutated,
            "gold_solution": row.get("gold_solution"),
            "aspect": "NumbersFactuality",
            "is_correct": False,
        }

        if len(reservoir) < sample_size:
            reservoir.append(sample)
        else:
            index = rng.randint(0, eligible - 1)
            if index < sample_size:
                reservoir[index] = sample

    if len(reservoir) < sample_size and not allow_smaller:
        raise ValueError(
            f"Collected {len(reservoir)} samples with numbers from {source_path} "
            f"(target={sample_size}, eligible={eligible}). Pass --allow-smaller to write fewer."
        )

    return reservoir, eligible


def parse_sizes(value: str) -> dict[str, int]:
    sizes = dict(DEFAULT_SAMPLE_SIZES)
    if not value:
        return sizes
    for part in value.split(","):
        split, raw_size = part.split("=", 1)
        split = split.strip()
        if split not in SPLITS:
            raise argparse.ArgumentTypeError(f"Unknown split in --sample-sizes: {split}")
        sizes[split] = int(raw_size)
    return sizes


def extend_split(
    split: str,
    *,
    dialog_dir: Path,
    dpo_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    allow_smaller: bool,
) -> None:
    if sample_size <= 0:
        print(f"{split}: skipped because sample size is {sample_size}")
        return

    dialog_path = dialog_dir / f"dialog_data_ft_{split}.json"
    base_dpo_path = dpo_dir / f"dpo_{split}.json"
    output_path = output_dir / f"dpo_{split}_extended.json"

    synthetic_pairs, eligible = sample_pairs_with_number_noise(
        dialog_path,
        sample_size=sample_size,
        seed=seed,
        allow_smaller=allow_smaller,
    )
    base_dpo = load_json_array(base_dpo_path)
    for row in base_dpo:
        row["is_correct"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows = base_dpo + synthetic_pairs
    output_path.write_text(json.dumps(write_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{split}: base={len(base_dpo)} synthetic={len(synthetic_pairs)} "
        f"eligible={eligible} wrote={len(write_rows)} -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialog-dir", type=Path, default=DEFAULT_DIALOG_DIR)
    parser.add_argument("--dpo-dir", type=Path, default=DEFAULT_DPO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DPO_DIR)
    parser.add_argument(
        "--sample-sizes",
        type=parse_sizes,
        default=dict(DEFAULT_SAMPLE_SIZES),
        help="Comma-separated split sizes, e.g. train=20000,dev=2000,test=2000. Use 0 to skip a split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; dev/test use +1/+2 unless --split-seeds is set.")
    parser.add_argument(
        "--split-seeds",
        default=None,
        help="Optional comma-separated seeds, e.g. train=42,dev=43,test=44.",
    )
    parser.add_argument("--allow-smaller", action="store_true")
    args = parser.parse_args()

    split_seeds = {split: args.seed + index for index, split in enumerate(SPLITS)}
    if args.split_seeds:
        for part in args.split_seeds.split(","):
            split, raw_seed = part.split("=", 1)
            split = split.strip()
            if split not in SPLITS:
                raise ValueError(f"Unknown split in --split-seeds: {split}")
            split_seeds[split] = int(raw_seed)
    else:
        split_seeds.update(DEFAULT_SEEDS)

    for split in SPLITS:
        extend_split(
            split,
            dialog_dir=args.dialog_dir,
            dpo_dir=args.dpo_dir,
            output_dir=args.output_dir,
            sample_size=args.sample_sizes[split],
            seed=split_seeds[split],
            allow_smaller=args.allow_smaller,
        )


if __name__ == "__main__":
    main()
