import pytest

from coding_agent.main import process_write_action_legacy


class _StubLLM:
    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        if agent_name == "syntax_check":
            return '{"valid": true}'
        if agent_name == "editor":
            return (
                "Renaming function.\n"
                "---FILE_START---\n"
                "function sendToAAAAA() { return 1 }\n"
                "sendToAAAAA()\n"
            )
        if agent_name == "validator":
            return '{"fixable": true, "cause": "Unknown", "suggestion": "Retry"}'
        return ""


@pytest.mark.asyncio
async def test_write_action_applies_editor_output_without_file_end(tmp_path):
    app = tmp_path / "app.js"
    app.write_text("function sendToPostHog() { return 1 }\nsendToPostHog()\n", encoding="utf-8")

    scan_result = {
        "action": "rename",
        "target": "sendToPostHog",
        "new_name": "sendToAAAAA",
        "files": [str(app)],
    }

    result = await process_write_action_legacy(_StubLLM(), scan_result)

    assert result["success"] is True
    assert str(app) in result.get("modified", [])
    updated = app.read_text(encoding="utf-8")
    assert "sendToAAAAA" in updated
    assert "sendToPostHog" not in updated


class _NoopEditorLLM:
    def __init__(self, original_cat: str):
        self._original_cat = original_cat

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        if agent_name == "syntax_check":
            return '{"valid": true}'
        if agent_name == "editor":
            return (
                "No-op edit.\n"
                "---FILE_START---\n"
                f"{self._original_cat}\n"
            )
        if agent_name == "validator":
            return '{"fixable": false, "cause": "No-op edit", "suggestion": "Change the file"}'
        return ""


@pytest.mark.asyncio
async def test_write_action_detects_no_changes(tmp_path):
    app = tmp_path / "app.js"
    original_source = "function sendToPostHog() { return 1 }\nsendToPostHog()\n"
    app.write_text(original_source, encoding="utf-8")

    original_cat = "1|function sendToPostHog() { return 1 }\n2|sendToPostHog()"
    llm = _NoopEditorLLM(original_cat)

    scan_result = {
        "action": "rename",
        "target": "sendToPostHog",
        "new_name": "sendToAAAAA",
        "files": [str(app)],
    }

    result = await process_write_action_legacy(llm, scan_result)

    assert result["success"] is False
    assert result.get("modified", []) == []
    assert app.read_text(encoding="utf-8") == original_source
