from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .channels.feishu import (
    FeishuAPIError,
    FeishuAppConfig,
    FeishuBotClient,
    FeishuChannel,
)
from .integrations.huggingface import (
    HFTrainerNotificationCallback,
    HFTrainerNotificationConfig,
)
from .integrations.training import TrainingNotifier
from .message import TextMessage
from .notifier import Notifier

try:
    __version__ = version("pjlab-cluster-notifier")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.0.0"

__all__ = [
    "FeishuAPIError",
    "FeishuAppConfig",
    "FeishuBotClient",
    "FeishuChannel",
    "HFTrainerNotificationCallback",
    "HFTrainerNotificationConfig",
    "Notifier",
    "TextMessage",
    "TrainingNotifier",
]
