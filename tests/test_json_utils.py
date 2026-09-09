from hy3scholar.json_utils import extract_json


def test_extract_fenced_json_with_braces_in_string() -> None:
    text = '说明如下：```json\n{"message":"x { y","score":0.8}\n```'
    assert extract_json(text) == {"message": "x { y", "score": 0.8}


def test_extract_json_after_reasoning_prefix() -> None:
    assert extract_json('分析完成。\n[1, {"ok": true}]\n结束') == [1, {"ok": True}]

