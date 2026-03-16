import json
import tempfile
from pathlib import Path

from coding_agent.tools import get_next_result_json_path


def test_get_next_result_json_path_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        p = get_next_result_json_path(d)
        assert p == Path(d).resolve() / "result.json"


def test_get_next_result_json_path_result_exists():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "result.json").write_text("{}")
        p = get_next_result_json_path(d)
        assert p == Path(d).resolve() / "result0.json"


def test_get_next_result_json_path_result0_exists():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "result.json").write_text("{}")
        (Path(d) / "result0.json").write_text("{}")
        p = get_next_result_json_path(d)
        assert p == Path(d).resolve() / "result1.json"


def test_result_json_payload_structure():
    payload = {
        "request": "Find usages of foo and save to JSON",
        "action": "search",
        "result": {"matches": [{"file": "app.js", "line": 1, "content": "foo()"}]},
    }
    with tempfile.TemporaryDirectory() as d:
        out_path = get_next_result_json_path(d)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(out_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["request"] == payload["request"]
        assert loaded["action"] == payload["action"]
        assert loaded["result"] == payload["result"]


def test_result_json_write_action_payload():
    payload = {
        "request": "Rename foo to bar",
        "action": "rename",
        "result": {"success": True, "modified": ["app.js"]},
    }
    with tempfile.TemporaryDirectory() as d:
        out_path = get_next_result_json_path(d)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["result"]["success"] is True
        assert loaded["result"]["modified"] == ["app.js"]
