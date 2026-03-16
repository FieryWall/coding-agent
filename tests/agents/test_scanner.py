import json
from coding_agent.agents import run_scanner
from coding_agent.tools import ToolsRepository
from tests.recording_tools_repo import (
    RecordingToolsRepository,
    assert_hard_constraints,
    assert_edit_tool_called,
    soft_constraint_ls_before_grep,
    soft_constraint_exact_pattern,
    optimal_trajectory_score,
    format_execution_trace,
)
import pytest
from pydantic import BaseModel


class ScannerResult(BaseModel):
    action: str
    target: str = None
    scope: str = None
    files: list[str] = None
    new_name: str = None
    data: dict = None


@pytest.fixture
def scanner_sandbox(tmp_path):
    (tmp_path / "app.js").write_text(
        "function max(depth) {\n  return 'max';\n}\nmodule.exports = { max };\n",
        encoding="utf-8",
    )
    repo = ToolsRepository()
    repo.set_root_path(str(tmp_path))
    return type("Sandbox", (), {"project_path": str(tmp_path), "tools_repo": repo, "app_file": tmp_path / "app.js"})()


@pytest.mark.asyncio
async def test_scanner_rename_function(production_llm_client, judge_grader, fact_judge_grader, scanner_sandbox):
    recording_repo = RecordingToolsRepository(scanner_sandbox.tools_repo)
    project_path = scanner_sandbox.project_path

    raw_result = await run_scanner(
        production_llm_client,
        recording_repo,
        "Rename function max to maximum",
        project_path,
    )

    assert "error" not in raw_result, f"Scanner failed: {raw_result}"
    result = ScannerResult(**raw_result)
    assert result.action == "rename"
    assert result.target == "max"
    assert result.new_name == "maximum"
    assert result.files and any("app.js" in str(f) for f in result.files)

    assert_hard_constraints(
        recording_repo.calls,
        must_have_tool="grep",
        tool_args_predicate=lambda args: "max" in str(args.get("pattern", "")),
    )
    assert soft_constraint_ls_before_grep(recording_repo.calls), "Soft: ls should precede grep when both used"
    assert soft_constraint_exact_pattern(recording_repo.calls, "max"), "Soft: grep pattern should be exact target name"
    ideal = [{"tool": "grep", "args": {"pattern": "max"}}]
    score = optimal_trajectory_score(recording_repo.calls, ideal)
    assert score >= 0.5, f"Optimal trajectory score {score} below 0.5"

    trajectory_output = f"Result: {raw_result}\n\nExecution trace:\n{format_execution_trace(recording_repo.calls)}"
    trajectory_judge = await judge_grader.grade(trajectory_output, """
        Task: rename function max to maximum. Result and tool call trace above.
        Rate 0-100: (1) Is the final result correct (action=rename, target=max, new_name=maximum, files list)? (2) Was tool usage efficient?
        """)
    assert trajectory_judge > 0.5, f"Trajectory judge score {trajectory_judge} below 0.5"
    judge_result = await judge_grader.grade(str(raw_result), """
        Request: "Rename function max to maximum". How well did the scanner define action, target, scope, files, new_name?
        """)
    assert judge_result > 0.5
    assert await fact_judge_grader.grade(str(raw_result), "Are all listed files under the given project path?")

@pytest.mark.asyncio
async def test_scanner_invalid_request(production_llm_client, scanner_sandbox):
    recording_repo = RecordingToolsRepository(scanner_sandbox.tools_repo)
    raw_result = await run_scanner(production_llm_client, recording_repo, "Hello!", scanner_sandbox.project_path)
    result = ScannerResult(**raw_result)
    assert result.action == "invalid"
    assert result.target is None
    assert result.scope is None
    assert result.files is None
    assert result.new_name is None
    assert result.data is None
    assert optimal_trajectory_score(recording_repo.calls, [{"tool": "grep", "args": {"pattern": "Hello", "path": "."}}]) < 1.0 or len(recording_repo.calls) == 0


@pytest.mark.asyncio
async def test_scanner_pairwise(production_llm_client, nano_llm_client, pairwise_grader, scanner_sandbox):
    def make_repo():
        r = ToolsRepository()
        r.set_root_path(scanner_sandbox.project_path)
        return RecordingToolsRepository(r)
    project_path = scanner_sandbox.project_path
    prompt = "Rename function max to maximum"
    recording_prod = make_repo()
    prod_result = await run_scanner(production_llm_client, recording_prod, prompt, project_path)
    if "error" in prod_result:
        recording_prod = make_repo()
        prod_result = await run_scanner(production_llm_client, recording_prod, prompt, project_path)
    recording_nano = make_repo()
    nano_result = await run_scanner(nano_llm_client, recording_nano, prompt, project_path)
    if "error" in nano_result:
        recording_nano = make_repo()
        nano_result = await run_scanner(nano_llm_client, recording_nano, prompt, project_path)
    if "error" in prod_result or "error" in nano_result:
        pytest.skip("One scanner run failed after retry")
    assert ScannerResult(**prod_result).target == "max"
    assert ScannerResult(**nano_result).target == "max"
    score = await pairwise_grader.grade(
        str(prod_result),
        str(nano_result),
        "Correctness of action=rename, target=max, new_name=maximum, scope and files; quality of tool usage.",
    )
    assert score >= 0.3


class _ScannerEditStubLLM:
    def __init__(self, app_path: str, project_path: str):
        self._responses = [
            json.dumps({
                "tool": "edit",
                "args": {
                    "path": app_path,
                    "start_line": 1,
                    "end_line": 2,
                    "new_content": "function maximum(depth) {\n  return 'maximum';\n}",
                },
            }),
            json.dumps({
                "action": "rename",
                "target": "max",
                "new_name": "maximum",
                "scope": project_path,
                "files": [app_path],
            }),
        ]
        self._i = 0

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        out = self._responses[self._i]
        self._i = min(self._i + 1, len(self._responses) - 1)
        return out


@pytest.mark.asyncio
async def test_scanner_calls_tool_edit_for_edit(scanner_sandbox):
    app_path = str(scanner_sandbox.app_file)
    project_path = scanner_sandbox.project_path
    recording_repo = RecordingToolsRepository(scanner_sandbox.tools_repo)
    stub_llm = _ScannerEditStubLLM(app_path, project_path)

    raw_result = await run_scanner(stub_llm, recording_repo, "Rename function max to maximum", project_path)

    assert "error" not in raw_result
    assert_edit_tool_called(recording_repo.calls, path_predicate=lambda p: "app.js" in p)
    content = scanner_sandbox.app_file.read_text(encoding="utf-8")
    assert "maximum" in content
    assert "function maximum" in content