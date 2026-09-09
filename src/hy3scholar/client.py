from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Protocol

import httpx

from .config import Settings
from .json_utils import extract_json


@dataclass(slots=True)
class Hy3Response:
    content: str
    reasoning_content: str = ""
    usage: dict[str, Any] | None = None
    warning: str | None = None
    search_results: list[dict[str, Any]] | None = None


class LLMClient(Protocol):
    def chat(
        self,
        *,
        system: str,
        user: str,
        reasoning_effort: str = "high",
        temperature: float | None = None,
        max_tokens: int | None = None,
        web_search: bool = False,
        search_source: str | None = None,
    ) -> Hy3Response: ...


class Hy3Client:
    """Tencent TokenHub OpenAI-compatible Hy3 client.

    The request intentionally sends ``reasoning_effort`` as a top-level field,
    matching TokenHub's Chat Completions contract.
    """

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        api_key = settings.require_api_key()
        self.http = httpx.Client(
            base_url=settings.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def chat(
        self,
        *,
        system: str,
        user: str,
        reasoning_effort: str = "high",
        temperature: float | None = None,
        max_tokens: int | None = None,
        web_search: bool = False,
        search_source: str | None = None,
    ) -> Hy3Response:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "top_p": self.settings.top_p,
            "reasoning_effort": reasoning_effort,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if web_search:
            selected_source = search_source or self.settings.web_search_source
            if selected_source not in {"lite", "standard"}:
                raise ValueError("search_source 只能是 lite 或 standard")
            payload["web_search_options"] = {
                "enable": True,
                "search_source": selected_source,
            }
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                response = self.http.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                warning = None
                if not content and reasoning:
                    content = reasoning
                    warning = (
                        "TokenHub 返回了空 content；本次兼容性回退使用 reasoning_content。"
                    )
                if not content.strip():
                    raise RuntimeError("Hy3 返回空内容")
                return Hy3Response(
                    content=content,
                    reasoning_content=reasoning,
                    usage=data.get("usage"),
                    warning=warning,
                    search_results=list(message.get("search_results") or []),
                )
            except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt + 1 >= self.settings.max_retries:
                    break
                time.sleep(min(8.0, (2**attempt) + random.random()))
        raise RuntimeError(f"Hy3 调用失败：{last_error}") from last_error

    def chat_json(self, **kwargs: Any) -> Any:
        return extract_json(self.chat(**kwargs).content)
