"""LangSmith Experiments for the "implement" action evals.

Run with:
    uv run pytest tests/agents/test_implement_langsmith.py -v -s

Results are visible in: LangSmith → Datasets & Experiments tab.
Each test maps to one persistent Dataset + one new Experiment per run.

Requires:
    LANGSMITH_API_KEY in environment (or .env)
    OPENAI_API_KEY in environment (or .env)
"""

import ast
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from langsmith import Client, aevaluate
from langsmith.schemas import Example, Run

from coding_agent.agents import run_editor_legacy


EXPERIMENT_PREFIX = "implement-evals"
EVAL_TIMEOUT_SEC = 60

BST_PLAYGROUND = Path(__file__).parents[2] / "playground" / "validate-binary-search-tree"

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


def _check_langsmith_configured():
    if not os.getenv("LANGSMITH_API_KEY"):
        pytest.skip("LANGSMITH_API_KEY not set — skipping LangSmith experiment")


def _ensure_dataset(client: Client, name: str, cases: list[dict]) -> str:
    """Return dataset name, creating it with examples if it doesn't exist yet."""
    datasets = list(client.list_datasets(dataset_name=name))
    if datasets:
        return name  # already exists — reuse across experiment runs

    dataset = client.create_dataset(name, description=f"Eval dataset: {name}")
    client.create_examples(
        inputs=[c["inputs"] for c in cases],
        outputs=[c.get("outputs", {}) for c in cases],
        dataset_id=dataset.id,
    )
    return name


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _model_id(llm_client) -> str:
    """Return a human-readable model identifier for LangSmith metadata."""
    if hasattr(llm_client, "model") and isinstance(llm_client.model, str):
        return llm_client.model                        # OpenAI: "gpt-4o-mini"
    if hasattr(llm_client, "model_path"):
        return Path(llm_client.model_path).name        # Local: "Qwen3-4B-..."
    return type(llm_client).__name__


# ---------------------------------------------------------------------------
# Experiment 1: Editor — implement new code in existing JS files
# ---------------------------------------------------------------------------

EDITOR_DATASET = "implement-evals/editor-legacy"

EDITOR_CASES = [
    {
        "inputs": {
            "task": "Add a function multiply(a, b) that returns a * b. Export it alongside add.",
            "file_path": "math.js",
            "file_content": "function add(a, b) {\n    return a + b;\n}\n\nmodule.exports = { add };\n",
        },
        "outputs": {
            "must_contain": ["multiply", "add"],
            "fact_question": (
                "Does the code define function multiply(a,b) returning a*b "
                "and export both add and multiply?"
            ),
        },
    },
    {
        "inputs": {
            "task": "Add a getName() method to the Luna class that returns the string 'luna'",
            "file_path": "luna.js",
            "file_content": (
                "class Luna {\n"
                "    call(depth = 0) {\n"
                "        if (depth >= 10) return 'luna';\n"
                "        return 'done';\n"
                "    }\n"
                "}\n\n"
                "const luna = (depth) => new Luna().call(depth);\n\n"
                "module.exports = { luna, Luna };\n"
            ),
        },
        "outputs": {
            "must_contain": ["getName", "class Luna"],
            "fact_question": (
                "Does the code contain a getName method inside the Luna class "
                "that returns 'luna', with the existing call method preserved?"
            ),
        },
    },
    {
        "inputs": {
            "task": (
                "Add a function getDepthLimit() that returns MAX_DEPTH. "
                "Export it alongside MAX_DEPTH."
            ),
            "file_path": "config.js",
            "file_content": "const MAX_DEPTH = 10;\n\nmodule.exports = { MAX_DEPTH };\n",
        },
        "outputs": {
            "must_contain": ["getDepthLimit", "MAX_DEPTH"],
            "fact_question": (
                "Does the code define getDepthLimit() returning MAX_DEPTH "
                "and export both MAX_DEPTH and getDepthLimit?"
            ),
        },
    },
]


@pytest.mark.asyncio
async def test_langsmith_editor_implement(production_llm_client, fact_judge_grader):
    """LangSmith Experiment: editor implements new code in existing JS files.

    Metrics pushed to LangSmith:
      - syntax_not_none   : 1.0 if editor returned non-None
      - contains_keywords : fraction of required identifiers present
      - llm_correctness   : 1.0 / 0.0 from FactJudgeGrader
    """
    _check_langsmith_configured()
    client = Client()
    dataset_name = _ensure_dataset(client, EDITOR_DATASET, EDITOR_CASES)

    local_scores: dict[str, list[float]] = {
        "syntax_not_none": [],
        "contains_keywords": [],
        "llm_correctness": [],
    }

    async def target(inputs: dict) -> dict:
        try:
            result = await asyncio.wait_for(
                run_editor_legacy(
                    production_llm_client,
                    inputs["task"],
                    inputs["file_path"],
                    inputs["file_content"],
                ),
                timeout=EVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            result = None
        return {"result": result}

    async def syntax_not_none(run: Run, example: Example) -> dict:
        score = 0.0 if run.outputs.get("result") is None else 1.0
        local_scores["syntax_not_none"].append(score)
        return {"key": "syntax_not_none", "score": score}

    async def contains_keywords(run: Run, example: Example) -> dict:
        result = run.outputs.get("result") or ""
        keywords = (example.outputs or {}).get("must_contain", [])
        found = sum(1 for kw in keywords if kw in result)
        score = found / len(keywords) if keywords else 1.0
        missing = [kw for kw in keywords if kw not in result]
        comment = f"Missing: {missing}" if missing else "All keywords present"
        local_scores["contains_keywords"].append(score)
        return {"key": "contains_keywords", "score": score, "comment": comment}

    async def llm_correctness(run: Run, example: Example) -> dict:
        result = run.outputs.get("result") or ""
        question = (example.outputs or {}).get("fact_question", "Is the code correct?")
        try:
            is_correct = await asyncio.wait_for(
                fact_judge_grader.grade(result, question),
                timeout=EVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            local_scores["llm_correctness"].append(0.0)
            return {"key": "llm_correctness", "score": 0.0, "comment": "TIMEOUT"}
        score = 1.0 if is_correct else 0.0
        local_scores["llm_correctness"].append(score)
        return {"key": "llm_correctness", "score": score, "comment": "YES" if is_correct else "NO"}

    await aevaluate(
        target,
        data=dataset_name,
        evaluators=[syntax_not_none, contains_keywords, llm_correctness],
        experiment_prefix=f"{EXPERIMENT_PREFIX}/{_model_id(production_llm_client)}",
        description="Editor (legacy) implements new functions and class methods in existing JS files.",
        metadata={"ls_model_name": _model_id(production_llm_client)},
        max_concurrency=1,
        client=client,
    )

    avg_correctness = _avg(local_scores["llm_correctness"])
    avg_keywords = _avg(local_scores["contains_keywords"])
    print(f"\nEditor implement experiment:")
    print(f"  llm_correctness:   {avg_correctness:.0%}  {local_scores['llm_correctness']}")
    print(f"  contains_keywords: {avg_keywords:.0%}  {local_scores['contains_keywords']}")

    assert avg_correctness >= 0.6, (
        f"LLM correctness {avg_correctness:.0%} below 60% threshold"
    )


# ---------------------------------------------------------------------------
# Experiment 2: BST — implement isValidBST, check syntax + runner pass rate
# ---------------------------------------------------------------------------

BST_DATASET = "implement-evals/bst-isValidBST"

BST_CASES = [
    {
        "inputs": {
            "task": (
                "Implement isValidBST method that checks if a binary tree is a valid BST. "
                "Use recursive approach with lower and upper bounds (float('-inf'), float('inf')). "
                "TreeNode has attributes: val, left, right. "
                "A valid BST requires all left subtree values strictly less than node, "
                "and all right subtree values strictly greater than node."
            ),
            "file_path": "solution.py",
            "file_content": BST_SOLUTION_TEMPLATE,
        },
        "outputs": {},
    },
]


@pytest.mark.asyncio
async def test_langsmith_bst_implement(production_llm_client):
    """LangSmith Experiment: editor implements isValidBST, verified by runner.

    Metrics pushed to LangSmith:
      - syntax_valid      : 1.0 if output is valid Python (ast.parse)
      - runner_pass_rate  : fraction of 15 LeetCode test cases that pass
    """
    _check_langsmith_configured()
    client = Client()
    dataset_name = _ensure_dataset(client, BST_DATASET, BST_CASES)

    local_scores: dict[str, list[float]] = {
        "syntax_valid": [],
        "runner_pass_rate": [],
    }

    async def target(inputs: dict) -> dict:
        try:
            result = await asyncio.wait_for(
                run_editor_legacy(
                    production_llm_client,
                    inputs["task"],
                    inputs["file_path"],
                    inputs["file_content"],
                ),
                timeout=EVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            result = None
        return {"result": result}

    async def syntax_valid(run: Run, example: Example) -> dict:
        result = run.outputs.get("result")
        if result is None:
            local_scores["syntax_valid"].append(0.0)
            return {"key": "syntax_valid", "score": 0.0, "comment": "Editor returned None"}
        try:
            ast.parse(result)
            local_scores["syntax_valid"].append(1.0)
            return {"key": "syntax_valid", "score": 1.0, "comment": "Valid Python"}
        except SyntaxError as e:
            local_scores["syntax_valid"].append(0.0)
            return {"key": "syntax_valid", "score": 0.0, "comment": f"SyntaxError: {e.msg} line {e.lineno}"}

    async def runner_pass_rate(run: Run, example: Example) -> dict:
        result = run.outputs.get("result")
        if result is None:
            local_scores["runner_pass_rate"].append(0.0)
            return {"key": "runner_pass_rate", "score": 0.0, "comment": "Editor returned None"}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy2(BST_PLAYGROUND / "runner.py", tmp_path / "runner.py")
            (tmp_path / "solution.py").write_text(result, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "runner.py", "--repeat", "1"],
                capture_output=True, text=True,
                cwd=str(tmp_path), timeout=30,
            )

        m = re.search(r"Results:\s*(\d+)/(\d+)\s*passed", proc.stdout)
        if not m:
            local_scores["runner_pass_rate"].append(0.0)
            return {
                "key": "runner_pass_rate",
                "score": 0.0,
                "comment": f"Could not parse runner output: {proc.stdout[:200]}",
            }
        passed, total = int(m.group(1)), int(m.group(2))
        score = passed / total if total else 0.0
        local_scores["runner_pass_rate"].append(score)
        return {
            "key": "runner_pass_rate",
            "score": score,
            "comment": f"{passed}/{total} test cases passed",
        }

    await aevaluate(
        target,
        data=dataset_name,
        evaluators=[syntax_valid, runner_pass_rate],
        experiment_prefix=f"{EXPERIMENT_PREFIX}/{_model_id(production_llm_client)}",
        description=(
            "Editor implements isValidBST in solution.py. "
            "Metrics: syntax_valid (Python AST), runner_pass_rate (15 LeetCode cases)."
        ),
        metadata={"ls_model_name": _model_id(production_llm_client)},
        max_concurrency=1,
        client=client,
    )

    sv = _avg(local_scores["syntax_valid"])
    rpr = _avg(local_scores["runner_pass_rate"])
    print(f"\nBST implement experiment:")
    print(f"  syntax_valid:      {sv:.0%}")
    print(f"  runner_pass_rate:  {rpr:.0%}")

    assert sv == 1.0, "BST implementation must be valid Python"
    assert rpr == 1.0, f"BST runner pass rate {rpr:.0%} — not all 15 test cases passed"
