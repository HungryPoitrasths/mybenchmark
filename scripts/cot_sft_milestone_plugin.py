"""MS-SWIFT callback that saves/evaluates at exact sample-exposure milestones."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from swift.callbacks import TrainerCallback, callbacks_map


def _positive_env(name: str) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def milestone_steps(
    *,
    train_count: int,
    global_batch: int,
    epochs: int,
    interval: int,
) -> dict[int, int]:
    steps_per_epoch = math.ceil(train_count / global_batch)
    result: dict[int, int] = {}
    total_exposures = train_count * epochs
    milestones = list(range(interval, total_exposures + 1, interval))
    if not milestones or milestones[-1] != total_exposures:
        milestones.append(total_exposures)
    for milestone in milestones:
        epoch_index, within_epoch = divmod(milestone - 1, train_count)
        exposure_in_epoch = within_epoch + 1
        step = epoch_index * steps_per_epoch + math.ceil(exposure_in_epoch / global_batch)
        result[step] = milestone
    return result


class CotSampleMilestoneCallback(TrainerCallback):

    def __init__(self, args: Any, trainer: Any):
        super().__init__(args, trainer)
        self.train_count = _positive_env("COT_SFT_TRAIN_COUNT")
        self.global_batch = _positive_env("COT_SFT_GLOBAL_BATCH")
        self.epochs = _positive_env("COT_SFT_EPOCHS")
        self.save_interval = _positive_env("COT_SFT_SAVE_EVERY_SAMPLES")
        self.eval_interval = _positive_env("COT_SFT_EVAL_EVERY_SAMPLES")
        self.log_interval = _positive_env("COT_SFT_LOG_EVERY_SAMPLES")
        self.save_milestones = milestone_steps(
            train_count=self.train_count,
            global_batch=self.global_batch,
            epochs=self.epochs,
            interval=self.save_interval,
        )
        self.eval_milestones = milestone_steps(
            train_count=self.train_count,
            global_batch=self.global_batch,
            epochs=self.epochs,
            interval=self.eval_interval,
        )
        self.log_milestones = milestone_steps(
            train_count=self.train_count,
            global_batch=self.global_batch,
            epochs=self.epochs,
            interval=self.log_interval,
        )

    def _samples_seen(self, step: int) -> int:
        steps_per_epoch = math.ceil(self.train_count / self.global_batch)
        completed_epochs, steps_in_epoch = divmod(step, steps_per_epoch)
        return completed_epochs * self.train_count + min(
            steps_in_epoch * self.global_batch,
            self.train_count,
        )

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.log_milestones:
            control.should_log = True
        if step in self.eval_milestones:
            control.should_evaluate = True
        if step in self.save_milestones:
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = int(state.global_step)
        target_samples = self.log_milestones.get(step)
        raw_logs = logs if isinstance(logs, dict) else {}
        loss = raw_logs.get("loss")
        if (
            target_samples is None
            or loss is None
            or not state.is_world_process_zero
        ):
            return control
        payload = {
            "schema_version": "predictive-spatial-cot-sft-train-loss-v1",
            "target_samples": target_samples,
            "samples_seen": self._samples_seen(step),
            "global_step": step,
            "loss": loss,
            "learning_rate": raw_logs.get("learning_rate"),
            "epoch": raw_logs.get("epoch"),
        }
        metrics_dir = Path(args.output_dir) / "training_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"samples_target_{target_samples:05d}.json"
        with metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"cot_sft_train_loss": payload}, ensure_ascii=False), flush=True)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = int(state.global_step)
        milestone = self.eval_milestones.get(step)
        if milestone is None or not state.is_world_process_zero:
            return control
        raw_metrics = metrics if isinstance(metrics, dict) else {}
        serializable_metrics = {
            str(key): value
            for key, value in raw_metrics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        payload = {
            "schema_version": "predictive-spatial-cot-sft-eval-v1",
            "milestone": milestone,
            "global_step": step,
            "samples_seen": self._samples_seen(step),
            "eval_loss": serializable_metrics.get("eval_loss"),
            "metrics": serializable_metrics,
        }
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / "eval_metrics.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"cot_sft_milestone_eval": payload}, ensure_ascii=False), flush=True)
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        milestone = self.save_milestones.get(step)
        if milestone is None or not state.is_world_process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "predictive-spatial-cot-sft-milestone-v1",
            "milestone": milestone,
            "global_step": step,
            "train_count": self.train_count,
            "global_batch": self.global_batch,
            "epochs": self.epochs,
            "interval": self.save_interval,
        }
        with (checkpoint_dir / "sample_milestone.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return control


callbacks_map["cot_sample_milestones"] = CotSampleMilestoneCallback
