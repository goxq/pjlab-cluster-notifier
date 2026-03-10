from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Any

from .channels.base import MessageChannel
from .message import TextMessage


def _is_channel(value: object) -> bool:
    return hasattr(value, "send") and callable(getattr(value, "send"))


def _normalize_channels(
    channels: MessageChannel | Iterable[MessageChannel],
) -> tuple[MessageChannel, ...]:
    if _is_channel(channels):
        return (channels,)  # type: ignore[arg-type]

    normalized = tuple(channels)
    if not normalized:
        raise ValueError("Notifier requires at least one channel")

    for channel in normalized:
        if not _is_channel(channel):
            raise TypeError("Each channel must implement send(message)")

    return normalized


class Notifier:
    def __init__(
        self,
        channels: MessageChannel | Iterable[MessageChannel],
        *,
        raise_on_channel_error: bool = False,
    ) -> None:
        self.channels = _normalize_channels(channels)
        self.raise_on_channel_error = raise_on_channel_error

    def send(self, message: TextMessage) -> list[Any | None]:
        responses: list[Any | None] = []
        for channel in self.channels:
            try:
                responses.append(channel.send(message))
            except Exception as error:
                if self.raise_on_channel_error:
                    raise
                print(
                    f"[pjnotifier] channel {channel.__class__.__name__} failed to send message: {error}",
                    file=sys.stderr,
                )
                responses.append(None)
        return responses

    def send_text(
        self,
        text: str,
        *,
        message_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[Any | None]:
        return self.send(
            TextMessage(
                text=text,
                message_id=message_id,
                metadata=metadata or {},
            )
        )
