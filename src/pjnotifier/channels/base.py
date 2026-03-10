from __future__ import annotations

from typing import Any, Protocol

from ..message import TextMessage


class MessageChannel(Protocol):
    def send(self, message: TextMessage) -> Any:
        ...
