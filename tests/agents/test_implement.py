"""TDD evals for the future "implement" action.

These tests define the expected behavior when the user asks the coding agent
to implement new functionality in existing files.

Pipeline:  Scanner (action=implement) → Change Planner (design) → Editor (code)

Tests are ordered by pipeline stage so that incremental progress is visible:
  Phase 1 – Scanner detects "implement"                      (tests 1-3)
  Phase 2 – Change Planner builds implementation design       (tests 4-6)
  Phase 3 – Editor writes correct code from instructions      (tests 7-10)

Expected to FAIL until each phase is implemented.
"""

import ast
import os
import shutil
import subprocess
import re
import pytest
from pydantic import BaseModel

from coding_agent.agents import (
    run_scanner,
    run_change_planner,
    run_editor,
    run_editor_legacy,
)
from coding_agent.tools import ToolsRepository
from coding_agent.vectorstore import RelevantChunk
from tests.recording_tools_repo import (
    RecordingToolsRepository,
    format_execution_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ScannerResult(BaseModel):
    action: str
    target: str = None
    scope: str = None
    files: list[str] = None
    new_name: str = None
    data: dict = None


# Sandbox with a small JS project for scanner tests
@pytest.fixture
def implement_sandbox(tmp_path):
    (tmp_path / "config.js").write_text(
        "const MAX_DEPTH = 10;\n\nmodule.exports = { MAX_DEPTH };\n",
        encoding="utf-8",
    )
    (tmp_path / "math.js").write_text(
        "function add(a, b) {\n    return a + b;\n}\n\n"
        "module.exports = { add };\n",
        encoding="utf-8",
    )
    (tmp_path / "luna.js").write_text(
        "class Luna {\n"
        "    call(depth = 0) {\n"
        "        if (depth >= 10) return 'luna';\n"
        "        return 'done';\n"
        "    }\n"
        "}\n\n"
        "const luna = (depth) => new Luna().call(depth);\n\n"
        "module.exports = { luna, Luna };\n",
        encoding="utf-8",
    )
    repo = ToolsRepository()
    repo.set_root_path(str(tmp_path))
    return type("Sandbox", (), {
        "project_path": str(tmp_path),
        "tools_repo": repo,
    })()


# File content strings for planner/editor tests (numbered lines like production)
CONFIG_JS = "1|const MAX_DEPTH = 10;\n2|\n3|module.exports = { MAX_DEPTH };\n"

MATH_JS = (
    "1|function add(a, b) {\n"
    "2|    return a + b;\n"
    "3|}\n"
    "4|\n"
    "5|module.exports = { add };\n"
)

LUNA_JS = (
    "1|class Luna {\n"
    "2|    call(depth = 0) {\n"
    "3|        if (depth >= 10) return 'luna';\n"
    "4|        return 'done';\n"
    "5|    }\n"
    "6|}\n"
    "7|\n"
    "8|const luna = (depth) => new Luna().call(depth);\n"
    "9|\n"
    "10|module.exports = { luna, Luna };\n"
)


# ===================================================================
# Phase 1: Scanner detects "implement" action
# ===================================================================

@pytest.mark.asyncio
async def test_scanner_recognizes_implement_action(production_llm_client, implement_sandbox):
    """Scanner should return action='implement' for new functionality requests."""
    recording_repo = RecordingToolsRepository(implement_sandbox.tools_repo)
    result = await run_scanner(
        production_llm_client,
        recording_repo,
        "Add a function multiply(a, b) that returns a * b to math.js",
        implement_sandbox.project_path,
    )
    assert "error" not in result, f"Scanner failed: {result}"
    parsed = ScannerResult(**result)
    assert parsed.action == "implement", (
        f"Expected action='implement', got '{parsed.action}'. "
        f"Scanner must distinguish 'implement new code' from rename/fix/search."
    )


@pytest.mark.asyncio
async def test_scanner_implement_finds_target_file(production_llm_client, implement_sandbox):
    """Scanner should identify the correct file for implementation."""
    recording_repo = RecordingToolsRepository(implement_sandbox.tools_repo)
    result = await run_scanner(
        production_llm_client,
        recording_repo,
        "Add a getName() method to the Luna class in luna.js",
        implement_sandbox.project_path,
    )
    assert "error" not in result, f"Scanner failed: {result}"
    parsed = ScannerResult(**result)
    assert parsed.action == "implement"
    assert parsed.files is not None, "Scanner must populate files list"
    assert any("luna.js" in f for f in parsed.files), (
        f"Expected luna.js in files, got {parsed.files}"
    )


@pytest.mark.asyncio
async def test_scanner_implement_uses_tools(production_llm_client, implement_sandbox):
    """Scanner should use tools (ls/grep/cat) to verify file existence."""
    recording_repo = RecordingToolsRepository(implement_sandbox.tools_repo)
    result = await run_scanner(
        production_llm_client,
        recording_repo,
        "Add a getDepthLimit function to config.js that returns MAX_DEPTH",
        implement_sandbox.project_path,
    )
    assert "error" not in result
    assert len(recording_repo.calls) >= 1, (
        f"Scanner should use at least one tool. Trace:\n"
        f"{format_execution_trace(recording_repo.calls)}"
    )


# ===================================================================
# Phase 2: Change Planner generates implementation design
# ===================================================================

@pytest.mark.asyncio
async def test_planner_produces_implement_plan(production_llm_client):
    """Change Planner should produce at least one active change for implement task."""
    changes = await run_change_planner(
        production_llm_client,
        task="Add a function multiply(a, b) that returns a * b. Export it alongside add.",
        file_path="math.js",
        file_content=MATH_JS,
        target="multiply",
    )
    assert changes is not None, "Planner must return changes (not None)"
    active = [c for c in changes if not c.skip]
    assert len(active) >= 1, (
        f"Planner must produce at least one active change. "
        f"Got {len(changes)} total, all skipped."
    )


@pytest.mark.asyncio
async def test_planner_implement_has_meaningful_instructions(
    production_llm_client, fact_judge_grader
):
    """Each planned change must have a clear instruction describing what to implement."""
    changes = await run_change_planner(
        production_llm_client,
        task="Add a function multiply(a, b) that returns a * b. Export it alongside add.",
        file_path="math.js",
        file_content=MATH_JS,
        target="multiply",
    )
    assert changes is not None
    active = [c for c in changes if not c.skip]
    assert len(active) >= 1

    plan_text = "\n".join(
        f"- original: {c.original!r}, instruction: {c.instruction!r}"
        for c in active
    )
    grade = await fact_judge_grader.grade(
        plan_text,
        "Does the plan describe adding a multiply function and exporting it?",
    )
    assert grade is True, f"Plan quality check failed. Plan:\n{plan_text}"


@pytest.mark.asyncio
async def test_planner_implement_class_method(production_llm_client, fact_judge_grader):
    """Planner should identify the correct insertion point inside a class."""
    changes = await run_change_planner(
        production_llm_client,
        task="Add a getName() method to the Luna class that returns the string 'luna'",
        file_path="luna.js",
        file_content=LUNA_JS,
        target="getName",
    )
    assert changes is not None
    active = [c for c in changes if not c.skip]
    assert len(active) >= 1

    plan_text = "\n".join(
        f"- original: {c.original!r}, instruction: {c.instruction!r}"
        for c in active
    )
    grade = await fact_judge_grader.grade(
        plan_text,
        "Does the plan describe adding a getName method to the Luna class "
        "that returns the string 'luna'?",
    )
    assert grade is True, f"Plan quality check failed. Plan:\n{plan_text}"


# ===================================================================
# Phase 3: Editor implements code from instructions
# ===================================================================

@pytest.mark.asyncio
async def test_editor_legacy_implements_new_function(
    production_llm_client, fact_judge_grader
):
    """Editor (legacy) should add a new function and export it."""
    content = "function add(a, b) {\n    return a + b;\n}\n\nmodule.exports = { add };\n"
    result = await run_editor_legacy(
        production_llm_client,
        "Add a function multiply(a, b) that returns a * b. Export it alongside add.",
        "math.js",
        content,
    )
    assert result is not None, "Editor returned None"
    assert "multiply" in result, f"Result must contain 'multiply'. Got:\n{result}"
    assert "add" in result, f"Result must preserve existing 'add'. Got:\n{result}"
    grade = await fact_judge_grader.grade(
        result,
        "Does the code define a function multiply(a, b) that returns a * b, "
        "and export both add and multiply?",
    )
    assert grade is True, f"Editor output quality check failed:\n{result}"


@pytest.mark.asyncio
async def test_editor_legacy_implements_class_method(
    production_llm_client, fact_judge_grader
):
    """Editor (legacy) should add a method to an existing class."""
    content = (
        "class Luna {\n"
        "    call(depth = 0) {\n"
        "        if (depth >= 10) return 'luna';\n"
        "        return 'done';\n"
        "    }\n"
        "}\n\n"
        "const luna = (depth) => new Luna().call(depth);\n\n"
        "module.exports = { luna, Luna };\n"
    )
    result = await run_editor_legacy(
        production_llm_client,
        "Add a getName() method to the Luna class that returns the string 'luna'",
        "luna.js",
        content,
    )
    assert result is not None, "Editor returned None"
    assert "getName" in result, f"Result must contain 'getName'. Got:\n{result}"
    assert "class Luna" in result, f"Result must preserve class. Got:\n{result}"
    grade = await fact_judge_grader.grade(
        result,
        "Does the code contain a getName method inside the Luna class "
        "that returns the string 'luna', with the existing call method preserved?",
    )
    assert grade is True, f"Editor output quality check failed:\n{result}"


@pytest.mark.asyncio
async def test_editor_chunk_implements_new_function(
    production_llm_client, fact_judge_grader
):
    """Editor (chunk-based) should implement a function given a code chunk."""
    chunk = RelevantChunk(
        content="function add(a, b) {\n    return a + b;\n}\n\nmodule.exports = { add };",
        file_path="math.js",
        start_line=1,
        end_line=5,
        chunk_type="module",
        name="math",
    )
    changes = await run_editor(
        production_llm_client,
        task="Add a function multiply(a, b) that returns a * b. Export it alongside add.",
        file_path="math.js",
        chunks=[chunk],
    )
    assert changes is not None, "Editor returned None"
    assert len(changes) >= 1, "Editor must produce at least one change"

    all_new = " ".join(c.new_content for c in changes)
    assert "multiply" in all_new, f"Changes must contain 'multiply'. Got: {changes}"
    grade = await fact_judge_grader.grade(
        str([{"original": c.original, "new_content": c.new_content} for c in changes]),
        "Do these changes add a multiply(a, b) function returning a * b "
        "and export it alongside add, without removing existing code?",
    )
    assert grade is True


@pytest.mark.asyncio
async def test_editor_implement_preserves_existing_code(
    production_llm_client, fact_judge_grader
):
    """Implementation must not break or remove existing functionality."""
    content = (
        "const MAX_DEPTH = 10;\n\n"
        "module.exports = { MAX_DEPTH };\n"
    )
    result = await run_editor_legacy(
        production_llm_client,
        "Add a function getDepthLimit() that returns MAX_DEPTH. "
        "Export it alongside MAX_DEPTH.",
        "config.js",
        content,
    )
    assert result is not None
    assert "MAX_DEPTH" in result, "Must preserve MAX_DEPTH constant"
    assert "getDepthLimit" in result, "Must contain new function"
    grade = await fact_judge_grader.grade(
        result,
        "Does the code (1) keep the original MAX_DEPTH = 10 constant, "
        "(2) define getDepthLimit() returning MAX_DEPTH, and "
        "(3) export both MAX_DEPTH and getDepthLimit?",
    )
    assert grade is True


# ===================================================================
# Pairwise: production vs nano model quality
# ===================================================================

@pytest.mark.asyncio
async def test_implement_editor_pairwise(
    production_llm_client, nano_llm_client, pairwise_grader
):
    """Production model should produce at least comparable implementation quality."""
    content = "function add(a, b) {\n    return a + b;\n}\n\nmodule.exports = { add };\n"
    task = "Add a function multiply(a, b) that returns a * b. Export it alongside add."

    prod_result = await run_editor_legacy(
        production_llm_client, task, "math.js", content
    )
    nano_result = await run_editor_legacy(
        nano_llm_client, task, "math.js", content
    )
    score = await pairwise_grader.grade(
        str(prod_result) if prod_result else "",
        str(nano_result) if nano_result else "",
        "Correctness of multiply implementation, preservation of add, "
        "proper exports, code style.",
    )
    assert score >= 0.3, f"Production model scored {score} vs nano (expected >= 0.3)"


# ===================================================================
# Phase 4: End-to-end — validate-binary-search-tree sandbox
# ===================================================================

BST_PLAYGROUND = os.path.join(
    os.path.dirname(__file__), "..", "..", "playground", "validate-binary-search-tree",
)

# Unimplemented template (what the editor receives)
BST_SOLUTION_TEMPLATE = (
    "from typing import Optional\n"
    "\n"
    "\n"
    "# Definition for a binary tree node.\n"
    "# class TreeNode:\n"
    "#     def __init__(self, val=0, left=None, right=None):\n"
    "#         self.val = val\n"
    "#         self.left = left\n"
    "#         self.right = right\n"
    "\n"
    "\n"
    "class Solution:\n"
    "    def isValidBST(self, root: Optional[TreeNode]) -> bool:\n"
    "        raise NotImplementedError\n"
)


@pytest.fixture
def bst_sandbox(tmp_path):
    """Copy validate-binary-search-tree project into a temp directory."""
    src = os.path.abspath(BST_PLAYGROUND)
    for name in ("runner.py",):
        shutil.copy2(os.path.join(src, name), tmp_path / name)
    # Write unimplemented solution template
    (tmp_path / "solution.py").write_text(BST_SOLUTION_TEMPLATE, encoding="utf-8")
    return tmp_path


def _parse_runner_results(stdout: str) -> dict:
    """Extract pass/total from runner output like 'Results: 14/15 passed  |  1 failed'."""
    m = re.search(r"Results:\s*(\d+)/(\d+)\s*passed\s*\|\s*(\d+)\s*failed", stdout)
    if not m:
        return {"passed": 0, "total": 0, "failed": 0, "pass_rate": 0.0}
    passed, total, failed = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return {
        "passed": passed,
        "total": total,
        "failed": failed,
        "pass_rate": passed / total if total else 0.0,
    }


@pytest.mark.asyncio
async def test_bst_implement_syntax_valid(production_llm_client):
    """Editor output must be syntactically valid Python."""
    result = await run_editor_legacy(
        production_llm_client,
        "Implement isValidBST method that checks if a binary tree is a valid BST. "
        "Use recursive approach with lower and upper bounds (float('-inf'), float('inf')). "
        "TreeNode has attributes: val, left, right. "
        "A valid BST has all left subtree values strictly less than node, "
        "and all right subtree values strictly greater than node.",
        "solution.py",
        BST_SOLUTION_TEMPLATE,
    )
    assert result is not None, "Editor returned None"

    # Syntax check: must parse as valid Python
    try:
        ast.parse(result)
    except SyntaxError as e:
        pytest.fail(
            f"Editor produced syntactically invalid Python:\n"
            f"  {e.msg} (line {e.lineno})\n\n{result}"
        )

    assert "isValidBST" in result, f"Must contain isValidBST method. Got:\n{result}"
    assert "class Solution" in result, f"Must preserve Solution class. Got:\n{result}"


@pytest.mark.asyncio
async def test_bst_implement_passes_runner(production_llm_client, bst_sandbox):
    """Editor implementation must pass the runner's test suite (15 cases)."""
    result = await run_editor_legacy(
        production_llm_client,
        "Implement isValidBST method that checks if a binary tree is a valid BST. "
        "Use recursive approach with lower and upper bounds (float('-inf'), float('inf')). "
        "TreeNode has attributes: val, left, right. "
        "A valid BST has all left subtree values strictly less than node, "
        "and all right subtree values strictly greater than node.",
        "solution.py",
        BST_SOLUTION_TEMPLATE,
    )
    assert result is not None, "Editor returned None"

    # Write editor output to sandbox
    (bst_sandbox / "solution.py").write_text(result, encoding="utf-8")

    # Run the test suite with repeat=1 for speed
    proc = subprocess.run(
        ["python", "runner.py", "--repeat", "1"],
        capture_output=True,
        text=True,
        cwd=str(bst_sandbox),
        timeout=30,
    )

    stats = _parse_runner_results(proc.stdout)
    print(f"\n--- BST Runner Output ---\n{proc.stdout}")
    if proc.stderr:
        print(f"--- stderr ---\n{proc.stderr}")

    assert stats["total"] == 15, (
        f"Runner should execute 15 test cases, got {stats['total']}. "
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert stats["pass_rate"] == 1.0, (
        f"BST implementation: {stats['passed']}/{stats['total']} passed "
        f"({stats['pass_rate']:.0%}), {stats['failed']} failed.\n"
        f"stdout:\n{proc.stdout}"
    )
