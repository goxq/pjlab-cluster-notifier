from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass

from ..message import TextMessage


DEFAULT_BASE_URL = "https://open.feishu.cn/open-apis"
DEFAULT_MESSAGE_STYLE = "interactive"


class FeishuAPIError(RuntimeError):
    pass


@dataclass(slots=True)
class FeishuAppConfig:
    app_id: str
    app_secret: str
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 10

    @classmethod
    def from_env(cls) -> "FeishuAppConfig":
        return cls(
            app_id=os.environ["FEISHU_APP_ID"],
            app_secret=os.environ["FEISHU_APP_SECRET"],
            base_url=os.getenv("FEISHU_BASE_URL", DEFAULT_BASE_URL),
            timeout=int(os.getenv("FEISHU_TIMEOUT", "10")),
        )


class FeishuBotClient:
    def __init__(self, config: FeishuAppConfig):
        self.config = config
        self._tenant_access_token: str | None = None
        self._token_expire_at = 0.0

    @classmethod
    def from_env(cls) -> "FeishuBotClient":
        return cls(FeishuAppConfig.from_env())

    def _post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                **(headers or {}),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw_response = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raw_response = error.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(raw_response)
            except json.JSONDecodeError:
                error_payload = raw_response or error.reason
            raise FeishuAPIError(
                f"request failed with HTTP {error.code}: {error_payload}"
            ) from error

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise FeishuAPIError(f"invalid JSON response: {raw_response}") from error

    def get_tenant_access_token(self, force_refresh: bool = False) -> str:
        now = time.time()
        if (
            not force_refresh
            and self._tenant_access_token
            and now < self._token_expire_at
        ):
            return self._tenant_access_token

        data = self._post_json(
            f"{self.config.base_url}/auth/v3/tenant_access_token/internal",
            {
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            },
        )
        if data.get("code") != 0:
            raise FeishuAPIError(f"get tenant_access_token failed: {data}")

        expire_seconds = int(data.get("expire", 7200))
        self._tenant_access_token = data["tenant_access_token"]
        self._token_expire_at = now + max(expire_seconds - 60, 0)
        return self._tenant_access_token

    def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: dict,
        message_id: str | None = None,
    ) -> dict:
        token = self.get_tenant_access_token()
        encoded_receive_id_type = urllib.parse.quote(receive_id_type, safe="")
        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
            "uuid": message_id or str(uuid.uuid4()),
        }
        data = self._post_json(
            f"{self.config.base_url}/im/v1/messages"
            f"?receive_id_type={encoded_receive_id_type}",
            payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        if data.get("code") != 0:
            raise FeishuAPIError(f"send message failed: {data}")
        return data

    def send_text(
        self,
        *,
        receive_id: str,
        text: str,
        receive_id_type: str = "chat_id",
        message_id: str | None = None,
    ) -> dict:
        return self.send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type="text",
            content={"text": text},
            message_id=message_id,
        )

    def send_post(
        self,
        *,
        receive_id: str,
        title: str,
        lines: list[str],
        receive_id_type: str = "chat_id",
        message_id: str | None = None,
    ) -> dict:
        return self.send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type="post",
            content={
                "zh_cn": {
                    "title": title,
                    "content": [
                        [{"tag": "text", "text": line}] for line in lines if line
                    ],
                }
            },
            message_id=message_id,
        )

    def send_interactive(
        self,
        *,
        receive_id: str,
        title: str,
        elements: list[dict],
        receive_id_type: str = "chat_id",
        message_id: str | None = None,
        template: str = "blue",
    ) -> dict:
        return self.send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type="interactive",
            content={
                "config": {
                    "wide_screen_mode": True,
                    "enable_forward": True,
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title,
                    },
                    "template": template,
                },
                "elements": elements,
            },
            message_id=message_id,
        )


@dataclass(slots=True)
class FeishuChannel:
    client: FeishuBotClient
    receive_id: str
    receive_id_type: str = "chat_id"
    message_style: str = DEFAULT_MESSAGE_STYLE

    @classmethod
    def from_env(cls) -> "FeishuChannel":
        return cls(
            client=FeishuBotClient.from_env(),
            receive_id=os.environ["FEISHU_RECEIVE_ID"],
            receive_id_type=os.getenv("FEISHU_RECEIVE_ID_TYPE", "open_id"),
            message_style=os.getenv(
                "FEISHU_MESSAGE_STYLE", DEFAULT_MESSAGE_STYLE
            ).lower(),
        )

    def send(self, message: TextMessage) -> dict:
        title, detail_lines = self._split_message_lines(message.text)

        if self.message_style in {"interactive", "card"}:
            return self.client.send_interactive(
                receive_id=self.receive_id,
                receive_id_type=self.receive_id_type,
                title=title,
                elements=self._build_interactive_elements(detail_lines),
                template=self._pick_template(title),
                message_id=message.message_id,
            )

        if self.message_style == "post":
            return self.client.send_post(
                receive_id=self.receive_id,
                receive_id_type=self.receive_id_type,
                title=title,
                lines=self._build_post_content(detail_lines),
                message_id=message.message_id,
            )

        return self.client.send_message(
            receive_id=self.receive_id,
            receive_id_type=self.receive_id_type,
            msg_type="text",
            content={"text": message.text},
            message_id=message.message_id,
        )

    def send_text(self, text: str, message_id: str | None = None) -> dict:
        return self.send(TextMessage(text=text, message_id=message_id))

    def _split_message_lines(self, text: str) -> tuple[str, list[str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "Notification", ["(empty)"]

        return lines[0], lines[1:] or ["(no details)"]

    def _build_post_content(self, lines: list[str]) -> list[str]:
        body: list[str] = []
        for key, value, raw_line in self._parse_detail_lines(lines):
            if key is None:
                body.append(f"- {raw_line}")
                continue
            body.append(f"{key}: {value}")

        return body or ["(no details)"]

    def _build_interactive_elements(self, lines: list[str]) -> list[dict]:
        detail_lines: list[str] = []
        compact_pairs: list[tuple[str, str]] = []
        summary_values: dict[str, str] = {}
        progress_line: str | None = None

        for key, value, raw_line in self._parse_detail_lines(lines):
            if key is None:
                detail_lines.append(self._escape_lark_md(raw_line))
                continue

            normalized_key = key.lower()
            if normalized_key == "progress":
                progress_line = value
                continue
            if normalized_key in _SUMMARY_DETAIL_LABELS:
                summary_values[normalized_key] = value
                continue
            if normalized_key in _SUMMARY_AUX_KEYS:
                summary_values[normalized_key] = value
                continue
            if normalized_key in _COMPACT_METRIC_LABELS:
                compact_pairs.append((_COMPACT_METRIC_LABELS[normalized_key], value))
                continue

            detail_lines.append(
                f"**{self._escape_lark_md(key)}**: {self._escape_lark_md(value)}"
            )

        elements: list[dict] = []
        summary_block = self._build_summary_block(progress_line, summary_values)
        if summary_block:
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "plain_text",
                        "content": summary_block,
                    },
                }
            )

        if compact_pairs:
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "plain_text",
                        "content": self._build_compact_metrics_block(compact_pairs),
                    },
                }
            )

        if detail_lines:
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "\n".join(detail_lines),
                    },
                }
            )
        else:
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "plain_text",
                        "content": "(no details)",
                    },
                }
            )

        return elements

    def _build_summary_block(
        self,
        progress_line: str | None,
        summary_values: dict[str, str],
    ) -> str:
        summary_pairs: list[tuple[str, str]] = []
        if progress_line:
            summary_pairs.append(("progress", progress_line))

        for key, label in _SUMMARY_DETAIL_LABELS.items():
            value = summary_values.get(key)
            if value:
                summary_pairs.append((label, value))

        dist_parts: list[str] = []
        node_count = summary_values.get("node_count")
        proc_per_node = summary_values.get("proc_per_node")
        node_rank = summary_values.get("node_rank")
        master_addr = summary_values.get("master_addr")
        if node_count and proc_per_node:
            dist_parts.append(f"{node_count}x{proc_per_node}")
        elif node_count:
            dist_parts.append(f"{node_count} nodes")
        elif proc_per_node:
            dist_parts.append(f"x{proc_per_node}")
        if node_rank:
            dist_parts.append(f"rank {node_rank}")
        if master_addr:
            dist_parts.append(f"master {master_addr}")
        if dist_parts:
            summary_pairs.append(("dist", " | ".join(dist_parts)))

        if not summary_pairs:
            return ""

        label_width = max(len(label) for label, _ in summary_pairs)
        return "\n".join(
            f"{label.ljust(label_width)} {value}" for label, value in summary_pairs
        )

    def _build_compact_metrics_block(self, pairs: list[tuple[str, str]]) -> str:
        columns = 2
        rows: list[str] = []
        formatted_pairs = [self._format_compact_pair(label, value) for label, value in pairs]
        left_column_width = max(
            (len(formatted_pairs[index]) for index in range(0, len(formatted_pairs), columns)),
            default=0,
        )

        for index in range(0, len(formatted_pairs), columns):
            row_pairs = formatted_pairs[index : index + columns]
            if len(row_pairs) == 2:
                rows.append(f"{row_pairs[0].ljust(left_column_width)}    {row_pairs[1]}")
            else:
                rows.append(row_pairs[0])

        return "\n".join(rows)

    def _format_compact_pair(self, label: str, value: str) -> str:
        normalized_value = value.replace(" samples/s", "/s")
        return f"{label:<4} {normalized_value}"

    def _parse_detail_lines(self, lines: list[str]) -> list[tuple[str | None, str | None, str]]:
        parsed: list[tuple[str | None, str | None, str]] = []
        key_value_pattern = re.compile(r"^([a-zA-Z0-9_\-. ]+):\s*(.+)$")
        for line in lines:
            matched = key_value_pattern.match(line)
            if not matched:
                parsed.append((None, None, line))
                continue
            key, value = matched.groups()
            parsed.append((key.strip(), value.strip(), line))

        return parsed or [(None, None, "(no details)")]

    def _pick_template(self, title: str) -> str:
        if title.startswith("✅"):
            return "green"
        if title.startswith("🚨"):
            return "red"
        if title.startswith("🧪"):
            return "orange"
        return "blue"

    def _escape_lark_md(self, text: str) -> str:
        escaped = text.replace("\\", "\\\\")
        for char in ("*", "_", "`", "[", "]", "(", ")"):
            escaped = escaped.replace(char, f"\\{char}")
        return escaped

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MESSAGE_STYLE",
    "FeishuAPIError",
    "FeishuAppConfig",
    "FeishuBotClient",
    "FeishuChannel",
]


_SUMMARY_DETAIL_LABELS = {
    "run": "run",
    "job_id": "job_id",
    "host": "host",
    "time": "time",
    "submit_time": "submit",
}

_SUMMARY_AUX_KEYS = {
    "master_addr",
    "node_count",
    "proc_per_node",
    "node_rank",
}

_COMPACT_METRIC_LABELS = {
    "stage": "stg",
    "epoch": "ep",
    "step": "step",
    "last_step": "step",
    "loss": "loss",
    "eval_loss": "eval",
    "accuracy": "acc",
    "learning_rate": "lr",
    "eta": "eta",
    "elapsed": "ela",
    "duration": "dur",
    "throughput": "tps",
    "grad_norm": "grad",
}
