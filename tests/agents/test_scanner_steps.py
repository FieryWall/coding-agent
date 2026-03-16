import json
import pytest
from coding_agent.agents import run_scanner_step
from coding_agent.agents.scanner import extract_nested_json, extract_action_json
from coding_agent.agents.utils import load_scanner_prompt


def test_step_gather_prompt_structure():
    prompt = load_scanner_prompt("gather", "Rename max to min", ".", "")
    assert "CURRENT STEP: GATHER" in prompt
    assert "Rename max to min" in prompt
    assert "grep" in prompt.lower()


def test_step_synthesize_prompt_structure():
    context = "[TOOL] grep result: {\"matches\": [{\"file\": \"./a.js\"}]}"
    prompt = load_scanner_prompt("synthesize", "Rename x to y", ".", context)
    assert "CURRENT STEP: SYNTHESIZE" in prompt
    assert context in prompt
    assert "files" in prompt.lower()


@pytest.mark.asyncio
@pytest.mark.eval
async def test_step_gather_outputs_tool_call(production_llm_client):
    response = await run_scanner_step(
        production_llm_client,
        "gather",
        "Rename function max to maximum",
        ".",
        context="",
    )
    tool_json = extract_nested_json(response, '"tool"')
    action_json = extract_action_json(response)
    assert tool_json or action_json, f"Expected tool or action JSON, got: {response[:200]}"
    if tool_json and "tool" in tool_json:
        assert tool_json["tool"] in ("ls", "grep", "cat")
        if tool_json["tool"] == "grep":
            assert "max" in str(tool_json.get("args", {}).get("pattern", ""))
    elif action_json:
        assert action_json.get("action") == "invalid"


class _StubGatherLLM:
    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        return '{"tool": "grep", "args": {"pattern": "max", "path": "."}}'


class _StubSynthesizeLLM:
    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        return '{"action": "rename", "target": "calculateTotal", "new_name": "computeSum", "scope": "./src", "files": ["./src/math.js", "./src/app.js", "./src/utils.js"], "data": {}}\n[DONE]'


@pytest.mark.asyncio
async def test_step_gather_with_stub():
    stub = _StubGatherLLM()
    response = await run_scanner_step(stub, "gather", "Rename max to min", ".", "")
    tool_json = extract_nested_json(response, '"tool"')
    assert tool_json and tool_json.get("tool") == "grep"
    assert "max" in str(tool_json.get("args", {}).get("pattern", ""))


@pytest.mark.asyncio
async def test_step_synthesize_with_stub():
    context = '[TOOL] grep result: {"matches": [{"file": "./src/math.js"}, {"file": "./src/app.js"}, {"file": "./src/utils.js"}], "total": 3}'
    stub = _StubSynthesizeLLM()
    response = await run_scanner_step(stub, "synthesize", "Rename calculateTotal to computeSum", "./src", context)
    action_json = extract_action_json(response)
    assert action_json and action_json.get("action") == "rename"
    assert len(action_json.get("files", [])) == 3


@pytest.mark.asyncio
@pytest.mark.eval
async def test_step_gather_outputs_tool_call(production_llm_client):
    response = await run_scanner_step(
        production_llm_client,
        "gather",
        "Rename function max to maximum",
        ".",
        context="",
    )
    tool_json = extract_nested_json(response, '"tool"')
    action_json = extract_action_json(response)
    assert tool_json or action_json, f"Expected tool or action JSON, got: {response[:200]}"
    if tool_json and "tool" in tool_json:
        assert tool_json["tool"] in ("ls", "grep", "cat")
        if tool_json["tool"] == "grep":
            assert "max" in str(tool_json.get("args", {}).get("pattern", ""))
    elif action_json:
        assert action_json.get("action") == "invalid"


@pytest.mark.asyncio
@pytest.mark.eval
async def test_step_synthesize_outputs_all_files(production_llm_client):
    context = """
[TOOL] grep result: {"pattern": "calculateTotal", "matches": [
    {"file": "./src/math.js", "line": 5, "content": "function calculateTotal()"},
    {"file": "./src/app.js", "line": 12, "content": "calculateTotal(items)"},
    {"file": "./src/utils.js", "line": 3, "content": "export { calculateTotal }"}
], "total": 3}
"""
    response = await run_scanner_step(
        production_llm_client,
        "synthesize",
        "Rename function calculateTotal to computeSum",
        "./src",
        context=context,
    )
    action_json = extract_action_json(response)
    assert action_json, f"Expected action JSON, got: {response[:300]}"
    assert action_json.get("action") == "rename"
    files = action_json.get("files") or []
    expected = ["./src/math.js", "./src/app.js", "./src/utils.js"]
    for f in expected:
        assert any(f in str(x) for x in files), f"Missing {f} in files: {files}"
