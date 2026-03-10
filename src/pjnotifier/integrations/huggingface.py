from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .training import TrainingNotifier

try:
    from transformers import TrainerCallback
except ImportError as exc:  # pragma: no cover - optional dependency
    _TRANSFORMERS_IMPORT_ERROR = exc

    class TrainerCallback:  # type: ignore[no-redef]
        pass

else:
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass(slots=True)
class HFTrainerNotificationConfig:
    notify_on_train_begin: bool = True
    notify_on_log: bool = True
    notify_on_evaluate: bool = True
    notify_on_save: bool = False
    notify_on_train_end: bool = True
    include_checkpoint_path: bool = True
    include_model_name: bool = True
    progress_every_n_steps: int | None = None
    progress_min_interval_seconds: float | None = None
    progress_on_first_step: bool = True
    progress_on_last_step: bool = True


class HFTrainerNotificationCallback(TrainerCallback):
    def __init__(
        self,
        training_notifier: TrainingNotifier,
        config: HFTrainerNotificationConfig | None = None,
    ) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required to use HFTrainerNotificationCallback"
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.training_notifier = training_notifier
        self.config = config or HFTrainerNotificationConfig()
        self._train_started_at: float | None = None
        self._last_progress_step: int | None = None
        self._last_progress_time: float | None = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self.config.notify_on_train_begin or not self._is_main_process(state):
            return control

        self._train_started_at = time.time()
        self.training_notifier.train_started(
            total_steps=self._normalize_int(getattr(state, "max_steps", None)),
            total_epochs=self._normalize_int(getattr(state, "num_train_epochs", None)),
            learning_rate=getattr(args, "learning_rate", None),
            model=self._infer_model_name(model) if self.config.include_model_name else None,
            dataset=None,
            extra=self._base_extra(args),
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.config.notify_on_log or not self._is_main_process(state):
            return control
        if not logs:
            return control
        if any(key.startswith("eval_") for key in logs):
            return control

        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        if loss is None and learning_rate is None:
            return control

        step = getattr(state, "global_step", None)
        total_steps = self._normalize_int(getattr(state, "max_steps", None))
        if not self._should_notify_progress(step, total_steps):
            return control
        eta_seconds = self._estimate_eta(step, total_steps)

        extra = self._sanitize_metrics(
            logs,
            exclude_keys={
                "loss",
                "learning_rate",
                "epoch",
                "step",
                "total_flos",
            },
        )
        self.training_notifier.train_progress(
            step=step or 0,
            total_steps=total_steps,
            epoch=getattr(state, "epoch", None),
            loss=loss,
            learning_rate=learning_rate,
            throughput=logs.get("train_steps_per_second"),
            eta_seconds=eta_seconds,
            extra=extra or None,
        )
        self._last_progress_step = step
        self._last_progress_time = time.time()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.config.notify_on_evaluate or not self._is_main_process(state):
            return control
        if not metrics:
            return control

        split = self._infer_split(metrics)
        self.training_notifier.eval_metrics(
            split=split,
            step=getattr(state, "global_step", None),
            epoch=getattr(state, "epoch", None),
            metrics=self._sanitize_metrics(metrics, exclude_keys={"epoch", "total_flos"}),
        )
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self.config.notify_on_save or not self._is_main_process(state):
            return control

        checkpoint_path = None
        if self.config.include_checkpoint_path:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        self.training_notifier.checkpoint_saved(
            path=checkpoint_path or args.output_dir,
            step=getattr(state, "global_step", None),
            metric=("best_metric", getattr(state, "best_metric", None)),
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.config.notify_on_train_end or not self._is_main_process(state):
            return control

        duration_seconds = None
        if self._train_started_at is not None:
            duration_seconds = max(time.time() - self._train_started_at, 0)

        best_metric = None
        if getattr(state, "best_metric", None) is not None:
            best_metric = ("metric", state.best_metric)

        extra = {}
        if getattr(state, "best_model_checkpoint", None):
            extra["best_model_checkpoint"] = state.best_model_checkpoint
        if getattr(state, "best_global_step", None) is not None:
            extra["best_global_step"] = state.best_global_step

        self.training_notifier.train_finished(
            duration_seconds=duration_seconds,
            best_metric=best_metric,
            extra=extra or None,
        )
        return control

    def _is_main_process(self, state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _infer_model_name(self, model: Any) -> str | None:
        if model is None:
            return None
        name_or_path = getattr(model, "name_or_path", None)
        if name_or_path:
            return str(name_or_path)
        config = getattr(model, "config", None)
        if config is not None:
            model_type = getattr(config, "_name_or_path", None) or getattr(config, "model_type", None)
            if model_type:
                return str(model_type)
        return model.__class__.__name__

    def _estimate_eta(self, step: int | None, total_steps: int | None) -> float | None:
        if not self._train_started_at or not step or not total_steps or step <= 0:
            return None
        if step >= total_steps:
            return 0.0
        elapsed = max(time.time() - self._train_started_at, 0.0)
        seconds_per_step = elapsed / step
        return seconds_per_step * (total_steps - step)

    def _should_notify_progress(
        self,
        step: int | None,
        total_steps: int | None,
    ) -> bool:
        step = self._normalize_int(step)
        if step is None:
            return False
        if self._last_progress_step == step:
            return False
        if self.config.progress_on_first_step and self._last_progress_step is None:
            return True
        if self.config.progress_on_last_step and total_steps is not None and step >= total_steps:
            return True
        if self.config.progress_every_n_steps is not None:
            every_n = max(int(self.config.progress_every_n_steps), 1)
            if step % every_n != 0:
                return False
        if self.config.progress_min_interval_seconds is not None and self._last_progress_time is not None:
            min_interval = max(float(self.config.progress_min_interval_seconds), 0.0)
            if time.time() - self._last_progress_time < min_interval:
                return False
        return True

    def _base_extra(self, args) -> Mapping[str, Any]:
        extra: dict[str, Any] = {}
        if getattr(args, "output_dir", None):
            extra["output_dir"] = args.output_dir
        if getattr(args, "per_device_train_batch_size", None) is not None:
            extra["per_device_batch_size"] = args.per_device_train_batch_size
        if getattr(args, "gradient_accumulation_steps", None) is not None:
            extra["grad_accumulation"] = args.gradient_accumulation_steps
        return extra

    def _infer_split(self, metrics: Mapping[str, Any]) -> str:
        for key in metrics:
            if key.startswith("eval_"):
                return "validation"
            if key.startswith("test_"):
                return "test"
        return "evaluation"

    def _sanitize_metrics(
        self,
        metrics: Mapping[str, Any],
        exclude_keys: set[str],
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in metrics.items():
            if key in exclude_keys or value is None:
                continue
            cleaned[key] = value
        return cleaned

    def _normalize_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return None
        return int_value if int_value > 0 else None
