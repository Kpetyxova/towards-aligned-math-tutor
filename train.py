"""Train the aligned math tutor with feedback SFT followed by dual-adapter DPO.

1. SFT on Educational Feedback Data.
2. Weighted DPO Alignment with a trainable policy adapter and frozen reference
   adapter loaded from the Stage 1 checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "training_sets"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "models"
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints"

VERSION_SFT = "add_correctness_flag_and_solution"
VERSION_DPO = "add_correctness_flag_and_solution_extended_dataset"

INPUT_VAR_CHOICES = ("v1", "v2", "v3", "v4")
INPUT_VAR_DESCRIPTIONS = {
    "v1": "dialog_context_only",
    "v2": "correctness_flag",
    "v3": "gold_solution",
    "v4": "correctness_flag_and_solution",
}

BASE_INSTRUCTION = (
    "You are a careful, precise, and supportive math tutor.\n"
    "Your task is to produce the next tutor response in an ongoing dialog."
)

GUIDELINES = (
    "Guidelines for your response:\n"
    "- If the student's solution is incorrect, guide the student toward the correct reasoning.\n"
    "- If the student's solution is correct, clearly acknowledge correctness and optionally provide brief reinforcement, intuition, or a natural next step.\n\n"
    "- Do NOT invent errors or suggest corrections if the solution is correct.\n"
)

ASPECT_WEIGHTS = {
    "Factuality": 1.0,
    "MistakeIdentification": 1.0,
    "Targetedness": 1.0,
    "RevealingAnswer": 1.0,
    "NumbersFactuality": 1.0,
    "Clarity": 0.5,
}


@dataclass(frozen=True)
class TrainingPaths:
    feedback_train: Path
    feedback_dev: Path
    dpo_train: Path
    dpo_dev: Path
    stage1_checkpoint: Path
    stage1_output: Path
    stage2_checkpoint: Path
    stage2_output: Path


def parse_input_var(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in INPUT_VAR_CHOICES:
        choices = ", ".join(INPUT_VAR_CHOICES)
        raise argparse.ArgumentTypeError(f"input_var must be one of: {choices}")
    return normalized


def build_instruction(input_var: str) -> str:
    fields = ["1) The dialog history between the student and the tutor."]
    next_index = 2
    if input_var in {"v2", "v4"}:
        fields.append(
            f"{next_index}) A boolean flag indicating whether the student's solution is mathematically correct."
        )
        next_index += 1
    if input_var in {"v3", "v4"}:
        fields.append(f"{next_index}) A gold solution to the task.")

    return BASE_INSTRUCTION + "\n\nYou will be given:\n" + "\n".join(fields) + "\n\n" + GUIDELINES


def sft_version_for_input_var(input_var: str) -> str:
    if input_var == "v4":
        return VERSION_SFT
    return f"{input_var}_{INPUT_VAR_DESCRIPTIONS[input_var]}"


def dpo_version_for_input_var(input_var: str) -> str:
    if input_var == "v4":
        return VERSION_DPO
    return f"{input_var}_{INPUT_VAR_DESCRIPTIONS[input_var]}_extended_dataset"


def resolve_cli_path(path: str | Path) -> Path:
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

    records: list[dict[str, Any]] = []
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


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def default_dpo_path(data_dir: Path, split: str, *, use_base_dpo: bool) -> Path:
    base = data_dir / "dpo" / f"dpo_{split}.json"
    extended = data_dir / "dpo" / f"dpo_{split}_extended.json"
    if not use_base_dpo and extended.exists():
        return extended
    return base


def build_paths(args: argparse.Namespace) -> TrainingPaths:
    data_dir = resolve_cli_path(args.data_dir)
    output_root = resolve_cli_path(args.output_root)
    checkpoint_root = resolve_cli_path(args.checkpoint_root)
    slug = model_slug(args.base_model)

    feedback_train = (
        resolve_cli_path(args.feedback_train_path)
        if args.feedback_train_path
        else data_dir / "dialog_data_ft" / "dialog_data_ft_train.json"
    )
    feedback_dev = (
        resolve_cli_path(args.feedback_dev_path)
        if args.feedback_dev_path
        else data_dir / "dialog_data_ft" / "dialog_data_ft_dev.json"
    )
    dpo_train = (
        resolve_cli_path(args.dpo_train_path)
        if args.dpo_train_path
        else default_dpo_path(data_dir, "train", use_base_dpo=args.use_base_dpo)
    )
    dpo_dev = (
        resolve_cli_path(args.dpo_dev_path)
        if args.dpo_dev_path
        else default_dpo_path(data_dir, "dev", use_base_dpo=args.use_base_dpo)
    )

    default_stage1_name = f"{slug}_l0_stage1_sft_feedback_lora_{sft_version_for_input_var(args.input_var)}"
    default_stage2_name = f"{slug}_l0_stage2_dpo_dual_adapter_lora_{dpo_version_for_input_var(args.input_var)}"

    return TrainingPaths(
        feedback_train=feedback_train,
        feedback_dev=feedback_dev,
        dpo_train=dpo_train,
        dpo_dev=dpo_dev,
        stage1_checkpoint=(
            resolve_cli_path(args.stage1_checkpoint_dir)
            if args.stage1_checkpoint_dir
            else checkpoint_root / default_stage1_name
        ),
        stage1_output=(
            resolve_cli_path(args.stage1_output)
            if args.stage1_output
            else output_root / default_stage1_name
        ),
        stage2_checkpoint=(
            resolve_cli_path(args.stage2_checkpoint_dir)
            if args.stage2_checkpoint_dir
            else checkpoint_root / default_stage2_name
        ),
        stage2_output=(
            resolve_cli_path(args.stage2_output)
            if args.stage2_output
            else output_root / default_stage2_name
        ),
    )


def next_item_has_same_dialog_id(items: list[dict[str, Any]], index: int) -> bool:
    current = items[index]
    if index + 1 >= len(items):
        return False
    nxt = items[index + 1]
    return current.get("dataset") == nxt.get("dataset") and str(current.get("id")) == str(nxt.get("id"))


def infer_is_correct(items: list[dict[str, Any]], index: int) -> bool:
    sample = items[index]
    existing = sample.get("is_correct")
    if isinstance(existing, bool):
        return existing
    if isinstance(existing, str) and existing.strip().lower() in {"true", "false"}:
        return existing.strip().lower() == "true"

    dataset = str(sample.get("dataset", "")).lower()
    sample_id = str(sample.get("id", "")).lower()

    if dataset.endswith("_correct") or dataset in {"mr_gsm8k_correct", "prm800k_correct"}:
        return True
    if dataset.endswith("_l0") or dataset in {"mr_gsm8k_l0", "prm800k_l0"}:
        return False
    if dataset == "socra_teach_single":
        return "incorrect" not in sample_id
    if dataset in {"socra_teach_multi", "mathdial"}:
        return not next_item_has_same_dialog_id(items, index)

    return False


def add_optional_prompt_inputs(prompt: str, *, input_var: str, is_correct: bool, gold_solution: Any) -> str:
    if input_var in {"v2", "v4"}:
        prompt += "\n\n[Student solution is correct]\n"
        prompt += str(is_correct)
    if input_var in {"v3", "v4"}:
        prompt += "\n\n[Gold solution to the task]\n"
        prompt += str(gold_solution).strip()
    return prompt


def build_feedback_sample(sample: dict[str, Any], is_correct: bool, *, input_var: str) -> dict[str, str]:
    prompt = build_instruction(input_var)
    prompt += "\n\n[Dialog History]\n"
    prompt += str(sample["dialog_history"]).strip()
    prompt = add_optional_prompt_inputs(
        prompt,
        input_var=input_var,
        is_correct=is_correct,
        gold_solution=sample["gold_solution"],
    )
    prompt += "\n\n[Next tutor response]\n"
    return {"prompt": prompt, "completion": str(sample["tutor_response"])}


def prepare_feedback_samples(
    path: Path,
    *,
    tokenizer: Any | None,
    max_samples: int | None,
    max_length: int,
    input_var: str,
) -> list[dict[str, str]]:
    items = read_json_or_jsonl(path)
    samples: list[dict[str, str]] = []
    skipped_for_length = 0

    for index, item in enumerate(items):
        if max_samples is not None and len(samples) >= max_samples:
            break
        processed = build_feedback_sample(item, infer_is_correct(items, index), input_var=input_var)
        if tokenizer is not None:
            text = processed["prompt"] + processed["completion"]
            if len(tokenizer.encode(text)) > max_length:
                skipped_for_length += 1
                continue
        samples.append(processed)

    if skipped_for_length:
        print(f"Skipped {skipped_for_length} feedback samples longer than {max_length} tokens.")
    return samples


def build_dpo_prompt(item: dict[str, Any], *, input_var: str) -> str:
    prompt = build_instruction(input_var)
    prompt += "\n\n[Dialog History]\n"
    prompt += str(item["dialog_history"]).strip()
    prompt = add_optional_prompt_inputs(
        prompt,
        input_var=input_var,
        is_correct=False,
        gold_solution=item["gold_solution"],
    )
    prompt += "\n\n[Next tutor response]\n\n"
    return prompt


def prepare_dpo_samples(
    path: Path,
    *,
    include_weights: bool,
    max_samples: int | None,
    input_var: str,
) -> list[dict[str, Any]]:
    items = read_json_or_jsonl(path)
    samples: list[dict[str, Any]] = []

    for item in items:
        if max_samples is not None and len(samples) >= max_samples:
            break
        chosen = item.get("preferred", item.get("chosen"))
        rejected = item.get("non_preferred", item.get("rejected"))
        if chosen is None or rejected is None:
            item_id = item.get("id", "<unknown>")
            raise ValueError(f"DPO item {item_id} in {path} is missing preferred/non_preferred text")

        sample: dict[str, Any] = {
            "prompt": build_dpo_prompt(item, input_var=input_var),
            "chosen": str(chosen),
            "rejected": str(rejected),
        }
        if include_weights:
            aspect = str(item.get("aspect", ""))
            sample["sample_weights"] = float(ASPECT_WEIGHTS.get(aspect, 1.0))
        samples.append(sample)

    return samples


def print_sample_preview(title: str, samples: list[dict[str, Any]]) -> None:
    print(f"\n{title}: {len(samples)} samples")
    if not samples:
        return
    sample = samples[0]
    for key in ("prompt", "completion", "chosen", "rejected"):
        if key in sample:
            value = str(sample[key])
            print(f"{key}: {value[:500]}{'...' if len(value) > 500 else ''}")
    if "sample_weights" in sample:
        print(f"sample_weights: {sample['sample_weights']}")


def dry_run(args: argparse.Namespace, paths: TrainingPaths) -> None:
    feedback_train_raw = read_json_or_jsonl(paths.feedback_train)
    feedback_dev_raw = read_json_or_jsonl(paths.feedback_dev)
    dpo_train_raw = read_json_or_jsonl(paths.dpo_train)
    dpo_dev_raw = read_json_or_jsonl(paths.dpo_dev)

    print("Resolved paths:")
    print(f"  input_var:      {args.input_var} ({INPUT_VAR_DESCRIPTIONS[args.input_var]})")
    print(f"  feedback train: {paths.feedback_train}")
    print(f"  feedback dev:   {paths.feedback_dev}")
    print(f"  DPO train:      {paths.dpo_train}")
    print(f"  DPO dev:        {paths.dpo_dev}")
    print(f"  Stage 1 output: {paths.stage1_output}")
    print(f"  Stage 2 output: {paths.stage2_output}")

    print("\nRaw data counts:")
    print(f"  feedback train: {len(feedback_train_raw)}")
    print(f"  feedback dev:   {len(feedback_dev_raw)}")
    print(f"  DPO train:      {len(dpo_train_raw)}")
    print(f"  DPO dev:        {len(dpo_dev_raw)}")

    print("\nFeedback train datasets:")
    for dataset, count in Counter(str(x.get("dataset", "")) for x in feedback_train_raw).most_common():
        print(f"  {dataset}: {count}")

    print("\nDPO train aspects:")
    for aspect, count in Counter(str(x.get("aspect", "")) for x in dpo_train_raw).most_common():
        print(f"  {aspect}: {count}")

    feedback_samples = prepare_feedback_samples(
        paths.feedback_train,
        tokenizer=None,
        max_samples=args.max_sft_samples or 2,
        max_length=args.sft_max_length,
        input_var=args.input_var,
    )
    dpo_samples = prepare_dpo_samples(
        paths.dpo_train,
        include_weights=True,
        max_samples=args.max_dpo_samples or 2,
        input_var=args.input_var,
    )
    print_sample_preview("Prepared feedback preview", feedback_samples)
    print_sample_preview("Prepared DPO preview", dpo_samples)
    print("\nDry run complete. Token-length filtering is only applied during real training.")


def import_training_dependencies() -> SimpleNamespace:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install the notebook dependencies "
            "(torch, transformers, datasets, peft, trl, wandb) before running training."
        ) from exc

    return SimpleNamespace(
        np=np,
        torch=torch,
        F=F,
        Dataset=Dataset,
        LoraConfig=LoraConfig,
        PeftModel=PeftModel,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        EarlyStoppingCallback=EarlyStoppingCallback,
        DPOConfig=DPOConfig,
        DPOTrainer=DPOTrainer,
        SFTConfig=SFTConfig,
        SFTTrainer=SFTTrainer,
    )


def seed_everything(seed: int, deps: SimpleNamespace) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    deps.np.random.seed(seed)
    deps.torch.manual_seed(seed)
    if deps.torch.cuda.is_available():
        deps.torch.cuda.manual_seed_all(seed)
    deps.torch.backends.cudnn.deterministic = True
    deps.torch.backends.cudnn.benchmark = False


def get_tokenizer(model_path: str | Path, deps: SimpleNamespace, *, trust_remote_code: bool) -> Any:
    kwargs = {"trust_remote_code": trust_remote_code}
    try:
        tokenizer = deps.AutoTokenizer.from_pretrained(model_path, use_fast=False, **kwargs)
    except Exception:
        tokenizer = deps.AutoTokenizer.from_pretrained(model_path, use_fast=True, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    tokenizer.padding_side = "right"
    return tokenizer


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


def get_lora_config(args: argparse.Namespace, deps: SimpleNamespace) -> Any:
    target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    return deps.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def make_config(config_cls: Any, **kwargs: Any) -> Any:
    """Small compatibility shim for renamed TrainerArguments fields."""
    pending = dict(kwargs)
    for _ in range(8):
        try:
            return config_cls(**pending)
        except TypeError as exc:
            message = str(exc)
            if "eval_strategy" in pending and "eval_strategy" in message:
                pending["evaluation_strategy"] = pending.pop("eval_strategy")
                continue
            match = re.search(r"unexpected keyword argument '([^']+)'", message)
            if match:
                bad_arg = match.group(1)
                if bad_arg in pending:
                    value = pending.pop(bad_arg)
                    print(f"Note: this installed training stack does not accept {bad_arg}={value!r}; dropping it.")
                    continue
            raise
    return config_cls(**pending)


def make_sft_trainer(deps: SimpleNamespace, **kwargs: Any) -> Any:
    try:
        return deps.SFTTrainer(**kwargs)
    except TypeError as exc:
        if "processing_class" not in kwargs or "processing_class" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs["tokenizer"] = kwargs.pop("processing_class")
        return deps.SFTTrainer(**kwargs)


def make_dpo_trainer_class(deps: SimpleNamespace) -> Any:
    torch = deps.torch
    F = deps.F
    DPOTrainer = deps.DPOTrainer

    class WeightedDPOTrainer(DPOTrainer):
        """DPOTrainer with exact per-sample weighted loss."""

        def get_batch_loss_metrics(self, model: Any, batch: dict[str, Any], train_eval: str = "train") -> tuple[Any, dict[str, float]]:
            sample_weights = batch.pop("sample_weights", None)
            metrics: dict[str, float] = {}
            prefix = "eval_" if train_eval == "eval" else ""

            forward_output = self.concatenated_forward(model, batch)
            if isinstance(forward_output, dict):
                policy_chosen_logps = forward_output.get("chosen_logps")
                if policy_chosen_logps is None:
                    policy_chosen_logps = forward_output.get("policy_chosen_logps")
                policy_rejected_logps = forward_output.get("rejected_logps")
                if policy_rejected_logps is None:
                    policy_rejected_logps = forward_output.get("policy_rejected_logps")
            elif isinstance(forward_output, (tuple, list)):
                if len(forward_output) >= 2:
                    policy_chosen_logps, policy_rejected_logps = forward_output[0], forward_output[1]
                else:
                    raise ValueError(f"Unexpected forward_output length: {len(forward_output)}")
            else:
                raise TypeError(f"Unexpected forward_output type: {type(forward_output)}")

            if policy_chosen_logps is None or policy_rejected_logps is None:
                raise KeyError("Could not read policy chosen/rejected log-probs from DPO forward output.")

            with torch.no_grad():
                if self.ref_model is None:
                    if hasattr(self, "null_ref_context"):
                        with self.null_ref_context():
                            ref_forward_output = self.concatenated_forward(model, batch)
                    elif hasattr(self, "get_ref_model"):
                        ref_forward_output = self.concatenated_forward(self.get_ref_model(), batch)
                    else:
                        ref_forward_output = self.concatenated_forward(model, batch)
                else:
                    ref_forward_output = self.concatenated_forward(self.ref_model, batch)

                if isinstance(ref_forward_output, dict):
                    ref_chosen_logps = ref_forward_output.get("chosen_logps")
                    if ref_chosen_logps is None:
                        ref_chosen_logps = ref_forward_output.get("policy_chosen_logps")
                    ref_rejected_logps = ref_forward_output.get("rejected_logps")
                    if ref_rejected_logps is None:
                        ref_rejected_logps = ref_forward_output.get("policy_rejected_logps")
                elif isinstance(ref_forward_output, (tuple, list)):
                    if len(ref_forward_output) >= 2:
                        ref_chosen_logps, ref_rejected_logps = ref_forward_output[0], ref_forward_output[1]
                    else:
                        raise ValueError(f"Unexpected ref_forward_output length: {len(ref_forward_output)}")
                else:
                    raise TypeError(f"Unexpected ref_forward_output type: {type(ref_forward_output)}")

            if ref_chosen_logps is None or ref_rejected_logps is None:
                raise KeyError("Could not read reference chosen/rejected log-probs from DPO forward output.")

            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = pi_logratios - ref_logratios

            loss_type = self.loss_type[0] if isinstance(self.loss_type, list) else self.loss_type
            if loss_type == "sigmoid":
                per_sample_losses = -F.logsigmoid(self.beta * logits)
            elif loss_type == "ipo":
                per_sample_losses = (logits - 1 / (2 * self.beta)) ** 2
            elif loss_type == "hinge":
                per_sample_losses = torch.relu(1 - self.beta * logits)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            def weighted_mean(x: Any, w: Any) -> Any:
                w32 = w.to(device=x.device, dtype=torch.float32)
                x32 = x.to(dtype=torch.float32)
                return (x32 * w32).sum() / w32.sum().clamp_min(1e-8)

            weights = None
            if sample_weights is not None:
                weights = sample_weights.to(per_sample_losses.device).float()
                loss = weighted_mean(per_sample_losses, weights)
                metrics[f"{prefix}sample_weights/mean"] = weights.mean().item()
                metrics[f"{prefix}sample_weights/min"] = weights.min().item()
                metrics[f"{prefix}sample_weights/max"] = weights.max().item()
                metrics[f"{prefix}sample_weights/sum"] = weights.sum().item()
            else:
                loss = per_sample_losses.mean()

            chosen_reg_alpha = float(getattr(self, "chosen_reg_alpha", 0.0) or 0.0)
            if chosen_reg_alpha > 0.0:
                chosen_reg_type = getattr(self, "chosen_reg_type", "hinge")
                chosen_reg_target = float(getattr(self, "chosen_reg_target", 0.0) or 0.0)
                chosen_reg_use_weights = bool(getattr(self, "chosen_reg_use_weights", True))
                chosen_reward_raw = self.beta * (policy_chosen_logps - ref_chosen_logps)
                if chosen_reg_type == "hinge":
                    reg_per_sample = torch.relu(chosen_reg_target - chosen_reward_raw)
                elif chosen_reg_type == "nll":
                    reg_per_sample = -policy_chosen_logps
                else:
                    raise ValueError(f"Unknown chosen_reg_type: {chosen_reg_type}")
                if weights is not None and chosen_reg_use_weights:
                    reg = weighted_mean(reg_per_sample, weights)
                else:
                    reg = reg_per_sample.to(dtype=torch.float32).mean()
                loss = loss + chosen_reg_alpha * reg
                metrics[f"{prefix}reg/chosen/{chosen_reg_type}"] = reg.detach().item()
                metrics[f"{prefix}reg/chosen/alpha"] = chosen_reg_alpha
                if chosen_reg_type == "hinge":
                    metrics[f"{prefix}reg/chosen/target"] = chosen_reg_target

            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
            reward_margins = chosen_rewards - rejected_rewards
            acc = (chosen_rewards > rejected_rewards).float()

            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
            metrics[f"{prefix}rewards/margins"] = reward_margins.mean().item()
            metrics[f"{prefix}rewards/accuracies"] = acc.mean().item()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()

            if weights is not None:
                metrics[f"{prefix}rewards/chosen_weighted"] = weighted_mean(chosen_rewards, weights).item()
                metrics[f"{prefix}rewards/rejected_weighted"] = weighted_mean(rejected_rewards, weights).item()
                metrics[f"{prefix}rewards/margins_weighted"] = weighted_mean(reward_margins, weights).item()
                metrics[f"{prefix}rewards/accuracies_weighted"] = weighted_mean(acc, weights).item()
                metrics[f"{prefix}logps/chosen_weighted"] = weighted_mean(policy_chosen_logps.detach(), weights).item()
                metrics[f"{prefix}logps/rejected_weighted"] = weighted_mean(policy_rejected_logps.detach(), weights).item()

            score_lambda = float(getattr(self, "score_lambda", 0.0) or 0.0)
            if score_lambda != 0.0:
                metrics[f"{prefix}rewards/score"] = (
                    metrics[f"{prefix}rewards/margins"] + score_lambda * metrics[f"{prefix}rewards/chosen"]
                )
                if weights is not None:
                    metrics[f"{prefix}rewards/score_weighted"] = (
                        metrics[f"{prefix}rewards/margins_weighted"]
                        + score_lambda * metrics[f"{prefix}rewards/chosen_weighted"]
                    )

            return loss, metrics

    return WeightedDPOTrainer


def make_dpo_trainer(deps: SimpleNamespace, trainer_cls: Any, **kwargs: Any) -> Any:
    try:
        return trainer_cls(**kwargs)
    except TypeError as exc:
        if "processing_class" not in kwargs or "processing_class" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs["tokenizer"] = kwargs.pop("processing_class")
        return trainer_cls(**kwargs)


def apply_wandb_settings(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    else:
        os.environ["WANDB_PROJECT"] = f"pedagogical-alignment-dual-{model_slug(args.base_model)}-l0-{args.input_var}"
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity


def report_to(args: argparse.Namespace) -> list[str] | str:
    return [] if args.no_wandb else "wandb"


def finish_wandb(args: argparse.Namespace) -> None:
    if args.no_wandb:
        return
    try:
        import wandb

        wandb.finish()
    except Exception:
        pass


def make_hf_dataset(deps: SimpleNamespace, samples: list[dict[str, Any]], *, seed: int) -> Any:
    if not samples:
        raise ValueError("Prepared dataset is empty.")
    return deps.Dataset.from_list(samples).shuffle(seed=seed)


def run_stage1(args: argparse.Namespace, paths: TrainingPaths, deps: SimpleNamespace) -> Any:
    print("\n" + "=" * 60)
    print("Stage 1: SFT on Educational Feedback Data")
    print("=" * 60)

    tokenizer = get_tokenizer(args.base_model, deps, trust_remote_code=args.trust_remote_code)
    train_samples = prepare_feedback_samples(
        paths.feedback_train,
        tokenizer=tokenizer,
        max_samples=args.max_sft_samples,
        max_length=args.sft_max_length,
        input_var=args.input_var,
    )
    dev_limit = args.max_eval_samples if args.max_eval_samples is not None else args.max_sft_eval_samples
    dev_samples = prepare_feedback_samples(
        paths.feedback_dev,
        tokenizer=tokenizer,
        max_samples=dev_limit,
        max_length=args.sft_max_length,
        input_var=args.input_var,
    )
    print(f"Feedback train: {len(train_samples)} | dev: {len(dev_samples)}")

    train_dataset = make_hf_dataset(deps, train_samples, seed=args.seed)
    eval_dataset = make_hf_dataset(deps, dev_samples, seed=args.seed)

    model_init_path = args.base_model
    model = deps.AutoModelForCausalLM.from_pretrained(
        model_init_path,
        torch_dtype=get_torch_dtype(args, deps),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    if args.sft_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    stage1_config = make_config(
        deps.SFTConfig,
        output_dir=str(paths.stage1_checkpoint),
        per_device_train_batch_size=args.sft_per_device_train_batch_size,
        gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
        learning_rate=args.sft_learning_rate,
        num_train_epochs=args.sft_epochs,
        logging_steps=args.sft_logging_steps,
        save_steps=args.sft_save_steps,
        save_strategy="steps",
        save_total_limit=args.sft_save_total_limit,
        bf16=deps.torch.cuda.is_available() and deps.torch.cuda.is_bf16_supported() and args.torch_dtype != "fp16",
        max_length=args.sft_max_length,
        warmup_ratio=args.sft_warmup_ratio,
        lr_scheduler_type="cosine",
        completion_only_loss=True,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=args.sft_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to(args),
        run_name=args.sft_run_name or f"stage1-sft-feedback-{args.input_var}_l0",
    )

    early_stopping = deps.EarlyStoppingCallback(
        early_stopping_patience=args.sft_early_stopping_patience,
        early_stopping_threshold=args.sft_early_stopping_threshold,
    )

    trainer = make_sft_trainer(
        deps,
        model=model,
        args=stage1_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping],
        peft_config=get_lora_config(args, deps),
        processing_class=tokenizer,
    )

    print(f"Output adapter: {paths.stage1_output}")
    trainer.train(resume_from_checkpoint=args.stage1_resume_from) if args.stage1_resume_from else trainer.train()
    trainer.save_model(str(paths.stage1_output))
    tokenizer.save_pretrained(str(paths.stage1_output))
    finish_wandb(args)

    del trainer
    del model
    gc.collect()
    if deps.torch.cuda.is_available():
        deps.torch.cuda.empty_cache()

    return tokenizer


def freeze_dual_adapter_model(model: Any) -> tuple[int, int]:
    model.set_adapter("reference")
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad = False
        elif "policy" in name and "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.set_adapter("policy")

    policy_trainable = sum(1 for name, param in model.named_parameters() if param.requires_grad and "policy" in name)
    reference_trainable = sum(1 for name, param in model.named_parameters() if param.requires_grad and "reference" in name)
    return policy_trainable, reference_trainable


def keep_sample_weights_in_collator(trainer: Any, deps: SimpleNamespace) -> None:
    original_collator = trainer.data_collator

    def collator_keep_weights(features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = original_collator(features)
        if features and "sample_weights" in features[0]:
            batch["sample_weights"] = deps.torch.tensor(
                [feature["sample_weights"] for feature in features],
                dtype=deps.torch.float32,
            )
        return batch

    trainer.data_collator = collator_keep_weights


def run_stage2_dpo(
    args: argparse.Namespace,
    paths: TrainingPaths,
    deps: SimpleNamespace,
    tokenizer: Any | None,
) -> None:
    print("\n" + "=" * 60)
    print("Stage 2: Weighted DPO Alignment with Dual Adapter")
    print("=" * 60)

    if not paths.stage1_output.exists():
        raise FileNotFoundError(
            f"Stage 1 adapter was not found at {paths.stage1_output}. "
            "Run Stage 1 first, or pass --stage1-output to an existing Stage 1 adapter."
        )

    if tokenizer is None:
        tokenizer_source = paths.stage1_output if paths.stage1_output.exists() else args.base_model
        tokenizer = get_tokenizer(tokenizer_source, deps, trust_remote_code=args.trust_remote_code)

    train_samples = prepare_dpo_samples(
        paths.dpo_train,
        include_weights=True,
        max_samples=args.max_dpo_samples,
        input_var=args.input_var,
    )
    dev_limit = args.max_eval_samples if args.max_eval_samples is not None else args.max_dpo_eval_samples
    dev_samples = prepare_dpo_samples(
        paths.dpo_dev,
        include_weights=True,
        max_samples=dev_limit,
        input_var=args.input_var,
    )
    print(f"DPO train: {len(train_samples)} | dev: {len(dev_samples)}")

    train_dataset = make_hf_dataset(deps, train_samples, seed=args.seed)
    eval_dataset = make_hf_dataset(deps, dev_samples, seed=args.seed)

    base_model = deps.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=get_torch_dtype(args, deps),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    policy_load_path = resolve_cli_path(args.stage2_resume_from) if args.stage2_resume_from else paths.stage1_output
    print(f"Policy adapter source: {policy_load_path}")
    print(f"Reference adapter source: {paths.stage1_output}")

    model = deps.PeftModel.from_pretrained(
        base_model,
        str(policy_load_path),
        is_trainable=True,
        adapter_name="policy",
    )
    model.load_adapter(str(paths.stage1_output), adapter_name="reference")
    policy_trainable, reference_trainable = freeze_dual_adapter_model(model)
    model.config.use_cache = False
    print(f"Trainable policy adapter tensors: {policy_trainable}")
    print(f"Trainable reference adapter tensors: {reference_trainable} (should be 0)")

    gradient_checkpointing_kwargs = {"use_reentrant": False} if args.dpo_gradient_checkpointing else None
    dpo_config = make_config(
        deps.DPOConfig,
        output_dir=str(paths.stage2_checkpoint),
        per_device_train_batch_size=args.dpo_per_device_train_batch_size,
        gradient_accumulation_steps=args.dpo_gradient_accumulation_steps,
        per_device_eval_batch_size=args.dpo_per_device_eval_batch_size,
        eval_accumulation_steps=args.dpo_eval_accumulation_steps,
        learning_rate=args.dpo_learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=args.dpo_epochs,
        logging_steps=args.dpo_logging_steps,
        save_steps=args.dpo_save_steps,
        save_strategy="steps",
        save_total_limit=args.dpo_save_total_limit,
        bf16=deps.torch.cuda.is_available() and deps.torch.cuda.is_bf16_supported() and args.torch_dtype != "fp16",
        beta=args.dpo_beta,
        loss_type="sigmoid",
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
        report_to=report_to(args),
        run_name=args.dpo_run_name or f"stage2-weighted-dpo-dual-adapter-{args.input_var}_l0",
        max_grad_norm=1.0,
        warmup_ratio=args.dpo_warmup_ratio,
        model_adapter_name="policy",
        ref_adapter_name="reference",
        gradient_checkpointing=args.dpo_gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        precompute_ref_log_probs=False,
        eval_strategy="steps",
        eval_steps=args.dpo_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/margins",
        greater_is_better=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    early_stopping = deps.EarlyStoppingCallback(
        early_stopping_patience=args.dpo_early_stopping_patience,
        early_stopping_threshold=args.dpo_early_stopping_threshold,
    )

    WeightedDPOTrainer = make_dpo_trainer_class(deps)
    trainer = make_dpo_trainer(
        deps,
        WeightedDPOTrainer,
        model=model,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping],
    )
    trainer.chosen_reg_type = args.dpo_chosen_reg_type
    trainer.chosen_reg_alpha = args.dpo_chosen_reg_alpha
    trainer.chosen_reg_target = args.dpo_chosen_reg_target
    trainer.chosen_reg_use_weights = True
    keep_sample_weights_in_collator(trainer, deps)

    print(f"Output adapter: {paths.stage2_output}")
    trainer.train(resume_from_checkpoint=args.stage2_resume_from) if args.stage2_resume_from else trainer.train()
    trainer.save_model(str(paths.stage2_output))
    tokenizer.save_pretrained(str(paths.stage2_output))
    finish_wandb(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train feedback SFT and weighted dual-adapter DPO for the aligned math tutor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--input-var",
        "--input_var",
        dest="input_var",
        type=parse_input_var,
        default="v4",
        help="Prompt input variant: v1 dialog only, v2 dialog + correctness flag, v3 dialog + gold solution, v4 dialog + both.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--feedback-train-path")
    parser.add_argument("--feedback-dev-path")
    parser.add_argument("--dpo-train-path")
    parser.add_argument("--dpo-dev-path")
    parser.add_argument("--use-base-dpo", action="store_true", help="Use dpo_train/dev.json instead of extended files.")
    parser.add_argument("--stage1-output")
    parser.add_argument("--stage2-output")
    parser.add_argument("--stage1-checkpoint-dir")
    parser.add_argument("--stage2-checkpoint-dir")
    parser.add_argument("--stage1-resume-from")
    parser.add_argument("--stage2-resume-from")
    parser.add_argument("--skip-stage1", action="store_true", help="Start from an existing Stage 1 adapter.")
    parser.add_argument("--only-stage1", action="store_true", help="Run SFT and stop before DPO.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and data formatting without loading a model.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")

    parser.add_argument("--max-sft-samples", type=int)
    parser.add_argument("--max-sft-eval-samples", type=int)
    parser.add_argument("--max-dpo-samples", type=int)
    parser.add_argument("--max-dpo-eval-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, help="Shared cap for both SFT and DPO dev sets.")

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")

    parser.add_argument("--sft-epochs", type=float, default=5)
    parser.add_argument("--sft-per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--sft-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-4)
    parser.add_argument("--sft-max-length", type=int, default=1280)
    parser.add_argument("--sft-logging-steps", type=int, default=50)
    parser.add_argument("--sft-save-steps", type=int, default=2000)
    parser.add_argument("--sft-save-total-limit", type=int, default=3)
    parser.add_argument("--sft-eval-steps", type=int, default=2000)
    parser.add_argument("--sft-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--sft-early-stopping-patience", type=int, default=4)
    parser.add_argument("--sft-early-stopping-threshold", type=float, default=0.01)
    parser.add_argument("--sft-run-name")
    parser.add_argument("--sft-gradient-checkpointing", action="store_true")

    parser.add_argument("--dpo-epochs", type=float, default=2)
    parser.add_argument("--dpo-per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--dpo-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dpo-per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--dpo-eval-accumulation-steps", type=int, default=8)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-max-length", type=int, default=768)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=512)
    parser.add_argument("--dpo-logging-steps", type=int, default=20)
    parser.add_argument("--dpo-save-steps", type=int, default=200)
    parser.add_argument("--dpo-save-total-limit", type=int, default=5)
    parser.add_argument("--dpo-eval-steps", type=int, default=200)
    parser.add_argument("--dpo-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dpo-beta", type=float, default=0.3)
    parser.add_argument("--dpo-chosen-reg-type", choices=["nll", "hinge"], default="nll")
    parser.add_argument("--dpo-chosen-reg-alpha", type=float, default=0.005)
    parser.add_argument("--dpo-chosen-reg-target", type=float, default=0.0)
    parser.add_argument("--dpo-early-stopping-patience", type=int, default=3)
    parser.add_argument("--dpo-early-stopping-threshold", type=float, default=0.1)
    parser.add_argument("--dpo-run-name")
    parser.add_argument("--no-dpo-gradient-checkpointing", dest="dpo_gradient_checkpointing", action="store_false")
    parser.set_defaults(dpo_gradient_checkpointing=True)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    paths = build_paths(args)
    apply_wandb_settings(args)

    if args.dry_run:
        dry_run(args, paths)
        return

    deps = import_training_dependencies()
    seed_everything(args.seed, deps)

    print(f"CUDA available: {deps.torch.cuda.is_available()}")
    if deps.torch.cuda.is_available():
        print(f"Visible GPU count: {deps.torch.cuda.device_count()}")
        print(f"BF16 supported: {deps.torch.cuda.is_bf16_supported()}")

    tokenizer = None
    if args.skip_stage1:
        if not paths.stage1_output.exists():
            raise FileNotFoundError(f"--skip-stage1 was set, but {paths.stage1_output} does not exist.")
    else:
        tokenizer = run_stage1(args, paths, deps)

    if args.only_stage1:
        print("Stopping after Stage 1 because --only-stage1 was set.")
        return

    run_stage2_dpo(args, paths, deps, tokenizer)
    print("\nTraining complete.")
    print(f"Stage 1 adapter: {paths.stage1_output}")
    print(f"Stage 2 DPO adapter: {paths.stage2_output}")


if __name__ == "__main__":
    main()
