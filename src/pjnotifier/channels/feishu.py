from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass

from ..message import TextMessage


DEFAULT_BASE_URL = "https://open.feishu.cn/open-apis"


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

        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

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


@dataclass(slots=True)
class FeishuChannel:
    client: FeishuBotClient
    receive_id: str
    receive_id_type: str = "chat_id"

    @classmethod
    def from_env(cls) -> "FeishuChannel":
        return cls(
            client=FeishuBotClient.from_env(),
            receive_id=os.environ["FEISHU_RECEIVE_ID"],
            receive_id_type=os.getenv("FEISHU_RECEIVE_ID_TYPE", "open_id"),
        )

    def send(self, message: TextMessage) -> dict:
        return self.client.send_message(
            receive_id=self.receive_id,
            receive_id_type=self.receive_id_type,
            msg_type="text",
            content={"text": message.text},
            message_id=message.message_id,
        )

    def send_text(self, text: str, message_id: str | None = None) -> dict:
        return self.send(TextMessage(text=text, message_id=message_id))

__all__ = [
    "DEFAULT_BASE_URL",
    "FeishuAPIError",
    "FeishuAppConfig",
    "FeishuBotClient",
    "FeishuChannel",
]
