from __future__ import annotations

import datetime as dt
import functools
import os
import socket
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, ParamSpec, Protocol, TypeVar

from ..channels.base import MessageChannel
from ..notifier import Notifier


P = ParamSpec("P")
R = TypeVar("R")


class TextDelivery(Protocol):
    def send_text(
        self,
        text: str,
        *,
        message_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass(slots=True)
class _TrainingState:
    started_at: float | None = None
    last_step: int | None = None
    total_steps: int | None = None
    last_epoch: float | None = None
    current_stage: str = "idle"


@dataclass(slots=True)
class TrainingNotifier:
    job_name: str
    channels: MessageChannel | Iterable[MessageChannel] | None = None
    delivery: TextDelivery | None = None
    run_name: str | None = None
    experiment_name: str | None = None
    host: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    default_extra: Mapping[str, Any] = field(default_factory=dict)
    raise_on_send_error: bool = False
    _delivery: TextDelivery = field(init=False, repr=False)
    _state: _TrainingState = field(default_factory=_TrainingState, init=False, repr=False)

    def __post_init__(self) -> None:
        self.default_extra = dict(self.default_extra)
        self._delivery = self._resolve_delivery()

    def _resolve_delivery(self) -> TextDelivery:
        has_channels = self.channels is not None
        has_delivery = self.delivery is not None
        if has_channels == has_delivery:
            raise ValueError("Provide exactly one of channels or delivery")
        if self.channels is not None:
            return Notifier(self.channels)
        assert self.delivery is not None
        return self.delivery

    def train_started(
        self,
        *,
        total_steps: int | None = None,
        total_epochs: int | None = None,
        learning_rate: float | None = None,
        model: str | None = None,
        dataset: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        self._state.started_at = time.time()
        self._state.total_steps = total_steps
        self._state.current_stage = "train"
        details = [
            ("stage", "train"),
            ("total_steps", total_steps),
            ("total_epochs", total_epochs),
            ("learning_rate", learning_rate),
            ("model", model),
            ("dataset", dataset),
        ]
        return self._send_event("🚀 Training Started", details, extra=extra)

    def train_progress(
        self,
        *,
        step: int,
        total_steps: int | None = None,
        epoch: float | None = None,
        loss: float | None = None,
        learning_rate: float | None = None,
        throughput: float | None = None,
        eta_seconds: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        total_steps = total_steps or self._state.total_steps
        self._state.last_step = step
        self._state.total_steps = total_steps
        self._state.last_epoch = epoch
        self._state.current_stage = "train"
        details = [
            ("stage", "train"),
            ("progress", self._format_progress(step, total_steps)),
            ("epoch", epoch),
            ("loss", loss),
            ("learning_rate", learning_rate),
            (
                "throughput",
                f"{self._format_value(throughput)} samples/s"
                if throughput is not None
                else None,
            ),
            ("eta", self._format_duration(eta_seconds) if eta_seconds is not None else None),
        ]
        return self._send_event("📈 Training Progress", details, extra=extra)

    def train_finished(
        self,
        *,
        duration_seconds: float | None = None,
        best_metric: tuple[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        self._state.current_stage = "finished"
        if duration_seconds is None and self._state.started_at is not None:
            duration_seconds = max(time.time() - self._state.started_at, 0)

        details = [
            ("stage", "train"),
            ("last_step", self._state.last_step),
            ("epoch", self._state.last_epoch),
            ("duration", self._format_duration(duration_seconds) if duration_seconds is not None else None),
        ]
        if best_metric is not None:
            metric_name, metric_value = best_metric
            details.append((f"best_{metric_name}", metric_value))
        return self._send_event("✅ Training Finished", details, extra=extra)

    def eval_started(
        self,
        *,
        split: str = "validation",
        step: int | None = None,
        epoch: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        self._state.current_stage = f"eval:{split}"
        if step is not None:
            self._state.last_step = step
        if epoch is not None:
            self._state.last_epoch = epoch
        details = [
            ("stage", f"eval:{split}"),
            ("step", step),
            ("epoch", epoch),
        ]
        return self._send_event("🧪 Eval Started", details, extra=extra)

    def eval_metrics(
        self,
        *,
        metrics: Mapping[str, Any],
        split: str = "validation",
        step: int | None = None,
        epoch: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        self._state.current_stage = f"eval:{split}"
        if step is not None:
            self._state.last_step = step
        if epoch is not None:
            self._state.last_epoch = epoch

        details: list[tuple[str, Any]] = [
            ("stage", f"eval:{split}"),
            ("step", step),
            ("epoch", epoch),
        ]
        details.extend((name, value) for name, value in metrics.items())
        return self._send_event("🧪 Eval Metrics", details, extra=extra)

    def checkpoint_saved(
        self,
        *,
        path: str,
        step: int | None = None,
        metric: tuple[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        if step is not None:
            self._state.last_step = step
        details = [
            ("stage", self._state.current_stage),
            ("step", step),
            ("checkpoint", path),
        ]
        if metric is not None:
            metric_name, metric_value = metric
            details.append((metric_name, metric_value))
        return self._send_event("💾 Checkpoint Saved", details, extra=extra)

    def train_failed(
        self,
        error: BaseException,
        *,
        stage: str = "train",
        include_traceback: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        self._state.current_stage = f"failed:{stage}"
        details: list[tuple[str, Any]] = [
            ("stage", stage),
            ("last_step", self._state.last_step),
            ("epoch", self._state.last_epoch),
            ("error", f"{type(error).__name__}: {error}"),
        ]
        if self._state.started_at is not None:
            details.append(("elapsed", self._format_duration(time.time() - self._state.started_at)))

        trace_extra: dict[str, Any] = {}
        if include_traceback:
            trace_extra["traceback"] = self._compact_traceback(error)
        if extra:
            trace_extra.update(extra)

        return self._send_event(
            "🚨 Training Failed",
            details,
            extra=trace_extra,
            suppress_exceptions=True,
        )

    def notify_on_failure(
        self,
        *,
        stage: str = "train",
        include_traceback: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    self.train_failed(
                        error,
                        stage=stage,
                        include_traceback=include_traceback,
                    )
                    raise

            return wrapper

        return decorator

    def _send_event(
        self,
        title: str,
        details: list[tuple[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
        suppress_exceptions: bool | None = None,
    ) -> Any:
        message = self._build_message(title, details, extra)
        return self._emit_text(message, suppress_exceptions=suppress_exceptions)

    def _emit_text(
        self,
        message: str,
        *,
        suppress_exceptions: bool | None = None,
    ) -> Any:
        if suppress_exceptions is None:
            suppress_exceptions = not self.raise_on_send_error

        try:
            return self._delivery.send_text(message)
        except Exception as error:
            if suppress_exceptions:
                print(f"[pjnotifier] failed to send notification: {error}", file=sys.stderr)
                return None
            raise

    def _build_message(
        self,
        title: str,
        details: list[tuple[str, Any]],
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        lines = [title]
        lines.extend(self._base_lines())
        lines.extend(self._detail_lines(details))
        if self.default_extra:
            lines.extend(self._detail_lines(list(self.default_extra.items())))
        if extra:
            lines.extend(self._detail_lines(list(extra.items())))
        lines.append(f"time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def _base_lines(self) -> list[str]:
        lines = [f"job: {self.job_name}"]
        if self.run_name:
            lines.append(f"run: {self.run_name}")
        if self.experiment_name:
            lines.append(f"experiment: {self.experiment_name}")
        lines.append(f"host: {self.host} pid={self.pid}")
        return lines

    def _detail_lines(self, details: list[tuple[str, Any]]) -> list[str]:
        lines: list[str] = []
        for key, value in details:
            if value is None or value == "":
                continue
            lines.append(f"{key}: {self._format_value(value)}")
        return lines

    def _format_progress(self, step: int, total_steps: int | None) -> str:
        if total_steps:
            percent = (step / total_steps) * 100
            return f"{step}/{total_steps} ({percent:.1f}%)"
        return str(step)

    def _compact_traceback(self, error: BaseException, max_chars: int = 1200) -> str:
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ).strip()
        if len(trace) <= max_chars:
            return trace
        return trace[: max_chars - 3] + "..."

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if value == 0:
                return "0"
            abs_value = abs(value)
            if abs_value >= 1000 or abs_value < 0.001:
                return f"{value:.4e}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        if isinstance(value, (list, tuple, set)):
            return ", ".join(self._format_value(item) for item in value)
        return str(value)

    def _format_duration(self, seconds: float | None) -> str:
        if seconds is None:
            return ""
        seconds_int = max(int(seconds), 0)
        hours, remainder = divmod(seconds_int, 3600)
        minutes, secs = divmod(remainder, 60)
        parts: list[str] = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)
