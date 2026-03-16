"""Eval tests: scanner correctly identifies action type for search, explain, fix."""
import pytest
from pydantic import BaseModel

from coding_agent.agents import run_scanner
from coding_agent.tools import ToolsRepository
from tests.recording_tools_repo import (
    RecordingToolsRepository,
    assert_hard_constraints,
    format_execution_trace,
)


class ScannerResult(BaseModel):
    action: str
    target: str = None
    scope: str = None
    files: list[str] = None
    new_name: str = None
    data: dict = None


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "utils.js").write_text(
        "function printStack(err) {\n"
        "  console.log(err.stack);\n"
        "}\n"
        "module.exports = { printStack };\n",
        encoding="utf-8",
    )
    (tmp_path / "app.js").write_text(
        "const { printStack } = require('./utils');\n"
        "function main() {\n"
        "  try { run(); } catch(e) { printStack(e); }\n"
        "}\n"
        "main();\n",
        encoding="utf-8",
    )
    repo = ToolsRepository()
    repo.set_root_path(str(tmp_path))
    return type("Sandbox", (), {"path": str(tmp_path), "tools_repo": repo})()


# --- search ---

@pytest.mark.asyncio
async def test_scanner_search_action(production_llm_client, judge_grader, sandbox):
    recording = RecordingToolsRepository(sandbox.tools_repo)
    raw = await run_scanner(
        production_llm_client, recording,
        "Find all usages of printStack",
        sandbox.path,
    )

    assert "error" not in raw, f"Scanner failed: {raw}"
    result = ScannerResult(**raw)
    assert result.action == "search", f"Expected action=search, got {result.action}"
    assert result.target == "printStack"
    assert result.new_name is None, "search should not set new_name"

    assert_hard_constraints(
        recording.calls,
        must_have_tool="grep",
        tool_args_predicate=lambda args: "printStack" in str(args.get("pattern", "")),
    )

    score = await judge_grader.grade(str(raw), """
        Request: "Find all usages of printStack".
        Did the scanner correctly set action=search, target=printStack, and list files containing it?
        """)
    assert score > 0.5, f"Judge score {score}"


# --- explain ---

@pytest.mark.asyncio
async def test_scanner_explain_action(production_llm_client, judge_grader, sandbox):
    recording = RecordingToolsRepository(sandbox.tools_repo)
    raw = await run_scanner(
        production_llm_client, recording,
        "What does the printStack function do?",
        sandbox.path,
    )

    assert "error" not in raw, f"Scanner failed: {raw}"
    result = ScannerResult(**raw)
    assert result.action == "explain", f"Expected action=explain, got {result.action}"
    assert result.target == "printStack"
    assert result.new_name is None, "explain should not set new_name"
    assert result.data and result.data.get("definition"), "explain should include data.definition"

    score = await judge_grader.grade(str(raw), """
        Request: "What does the printStack function do?"
        Did the scanner correctly set action=explain, target=printStack, and provide a definition?
        """)
    assert score > 0.5, f"Judge score {score}"


# --- fix ---

@pytest.fixture
def fix_sandbox(tmp_path):
    (tmp_path / "broken.js").write_text(
        "function processData(items) {\n"
        "  for (let i = 0; i <= items.length; i++) {\n"
        "    console.log(items[i].name);\n"
        "  }\n"
        "}\n"
        "processData([{name: 'a'}, {name: 'b'}]);\n",
        encoding="utf-8",
    )
    repo = ToolsRepository()
    repo.set_root_path(str(tmp_path))
    return type("Sandbox", (), {"path": str(tmp_path), "tools_repo": repo})()


@pytest.mark.asyncio
async def test_scanner_fix_action(production_llm_client, judge_grader, fix_sandbox):
    recording = RecordingToolsRepository(fix_sandbox.tools_repo)
    raw = await run_scanner(
        production_llm_client, recording,
        "Fix the bug in processData function",
        fix_sandbox.path,
    )

    assert "error" not in raw, f"Scanner failed: {raw}"
    result = ScannerResult(**raw)
    assert result.action == "fix", f"Expected action=fix, got {result.action}"
    assert result.target is not None, "fix should have a target"
    assert result.new_name is None, "fix should not set new_name"
    assert result.files, "fix should list files to modify"

    assert_hard_constraints(
        recording.calls,
        must_have_tool="grep",
        tool_args_predicate=lambda args: "processData" in str(args.get("pattern", "")),
    )

    score = await judge_grader.grade(str(raw), """
        Request: "Fix the bug in processData function".
        Did the scanner correctly set action=fix (not rename!), target=processData, and list the file?
        """)
    assert score > 0.5, f"Judge score {score}"
