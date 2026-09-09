import json

import httpx

from hy3scholar.client import Hy3Client
from hy3scholar.config import Settings


def test_hy3_tokenhub_request_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer secret-test"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ok", "reasoning_content": "hidden"}}
                ],
                "usage": {"total_tokens": 10},
            },
        )

    settings = Settings(
        api_key="secret-test",
        base_url="https://tokenhub.tencentmaas.com/v1",
        model="hy3",
        max_retries=1,
    )
    client = Hy3Client(settings, transport=httpx.MockTransport(handler))
    response = client.chat(system="s", user="u", reasoning_effort="high")
    client.close()
    assert response.content == "ok"
    assert captured["model"] == "hy3"
    assert captured["reasoning_effort"] == "high"
    assert captured["stream"] is False


def test_reasoning_content_compatibility_fallback() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "reasoning_content": "x"}}]},
        )

    client = Hy3Client(
        Settings(api_key="x", max_retries=1), transport=httpx.MockTransport(handler)
    )
    response = client.chat(system="s", user="u")
    client.close()
    assert response.content == "x"
    assert response.warning


def test_hy3_web_search_contract_and_sources() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "found",
                            "search_results": [
                                {
                                    "index": 1,
                                    "url": "https://arxiv.org/abs/1706.03762",
                                    "name": "Attention Is All You Need",
                                    "snippet": "Transformer paper",
                                    "site": "arXiv",
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = Hy3Client(
        Settings(api_key="x", max_retries=1),
        transport=httpx.MockTransport(handler),
    )
    response = client.chat(system="s", user="u", web_search=True)
    client.close()
    assert captured["web_search_options"] == {
        "enable": True,
        "search_source": "lite",
    }
    assert response.search_results[0]["site"] == "arXiv"
