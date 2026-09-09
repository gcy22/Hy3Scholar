from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Parse JSON from a raw or fenced Hy3 response.

    The scanner uses JSONDecoder.raw_decode so braces inside quoted strings do not
    break extraction.
    """

    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.I | re.S)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(stripped)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"[\[{]", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("Hy3 响应中没有可解析的 JSON")

