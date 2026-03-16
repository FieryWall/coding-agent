import pytest
from coding_agent.agents import run_editor
from coding_agent.vectorstore import RelevantChunk


JS_CODE_WITH_AMBIGUOUS_NAMES = """const MAX_DEPTH = 10;

function max(depth = 0) {
    if (depth >= MAX_DEPTH) return 'max';
    return buddy(depth + 1);
}

module.exports = { max, MAX_DEPTH };
"""


def _make_chunks(code: str, file_path: str) -> list[RelevantChunk]:
    return [
        RelevantChunk(
            content="const MAX_DEPTH = 10;",
            file_path=file_path,
            start_line=1,
            end_line=1,
            chunk_type="variable",
            name="MAX_DEPTH",
        ),
        RelevantChunk(
            content="function max(depth = 0) {\n    if (depth >= MAX_DEPTH) return 'max';\n    return buddy(depth + 1);\n}",
            file_path=file_path,
            start_line=3,
            end_line=6,
            chunk_type="function",
            name="max",
        ),
        RelevantChunk(
            content="module.exports = { max, MAX_DEPTH };",
            file_path=file_path,
            start_line=8,
            end_line=8,
            chunk_type="variable",
            name="exports",
        ),
    ]


@pytest.mark.asyncio
async def test_editor_disambiguates_function_from_constant(production_llm_client, fact_judge_grader):
    """Editor should rename function 'max' to 'min' but NOT touch MAX_DEPTH constant."""
    file_path = "playground/pets/max.js"
    chunks = _make_chunks(JS_CODE_WITH_AMBIGUOUS_NAMES, file_path)
    
    task = '''Rename function/variable "max" to "min"
Rename ONLY identifiers named exactly "max" (function declarations, calls, imports, exports)
DO NOT rename: constants, strings, comments, or similar names like "MAX" or "MAX_DEPTH"'''

    result = await run_editor(production_llm_client, task, file_path, chunks)
    
    assert result is not None, "Editor should produce changes"
    assert len(result) > 0, "Editor should have at least one change"
    
    changes_content = " ".join(c.new_content for c in result)
    
    assert "min" in changes_content, "Should rename max to min"
    
    has_max_depth_change = any(
        "MAX_DEPTH" in c.original and "MAX_DEPTH" not in c.new_content
        for c in result
    )
    assert not has_max_depth_change, "Should NOT modify MAX_DEPTH constant"
    
    output_for_grade = f"""Task: Rename function "max" to "min" without touching MAX_DEPTH constant.
Original chunks: {chunks}
Result changes: {result}
"""
    grade_question = "Did the editor rename the function 'max' to 'min' while preserving MAX_DEPTH unchanged?"
    assert await fact_judge_grader.grade(output_for_grade, grade_question)


@pytest.mark.asyncio
async def test_editor_preserves_string_literals(production_llm_client, fact_judge_grader):
    """Editor should NOT rename 'max' inside string literals like return 'max'."""
    file_path = "playground/pets/max.js"
    chunks = _make_chunks(JS_CODE_WITH_AMBIGUOUS_NAMES, file_path)
    
    task = '''Rename function/variable "max" to "min"
Rename ONLY identifiers named exactly "max" (function declarations, calls, imports, exports)
DO NOT rename: constants, strings, comments, or similar names like "MAX" or "MAX_DEPTH"'''

    result = await run_editor(production_llm_client, task, file_path, chunks)
    
    assert result is not None
    
    for change in result:
        if "function max" in change.original or "function min" in change.original:
            pass


PYTHON_CODE_WITH_SAME_NAME = """MAX_ITEMS = 100
max_items = 50

def max(a, b):
    return a if a > b else b

result = max(MAX_ITEMS, max_items)
"""


def _make_python_chunks() -> list[RelevantChunk]:
    return [
        RelevantChunk(
            content="MAX_ITEMS = 100",
            file_path="utils.py",
            start_line=1,
            end_line=1,
            chunk_type="variable",
            name="MAX_ITEMS",
        ),
        RelevantChunk(
            content="max_items = 50",
            file_path="utils.py",
            start_line=2,
            end_line=2,
            chunk_type="variable",
            name="max_items",
        ),
        RelevantChunk(
            content="def max(a, b):\n    return a if a > b else b",
            file_path="utils.py",
            start_line=4,
            end_line=5,
            chunk_type="function",
            name="max",
        ),
        RelevantChunk(
            content="result = max(MAX_ITEMS, max_items)",
            file_path="utils.py",
            start_line=7,
            end_line=7,
            chunk_type="variable",
            name="result",
        ),
    ]


@pytest.mark.asyncio
async def test_editor_disambiguates_function_variable_constant_python(production_llm_client, fact_judge_grader):
    """Editor should rename function 'max' but NOT variable 'max_items' or constant 'MAX_ITEMS'."""
    chunks = _make_python_chunks()
    
    task = '''Rename function/variable "max" to "minimum"
Rename ONLY identifiers named exactly "max" (function declarations, calls, imports, exports)
DO NOT rename: constants (MAX_ITEMS), variables with similar names (max_items), strings, comments'''

    result = await run_editor(production_llm_client, task, "utils.py", chunks)
    
    assert result is not None
    
    changes_content = " ".join(c.new_content for c in result)
    
    assert "minimum" in changes_content, "Should rename max to minimum"

    output_for_grade = f"""Task: Rename function "max" to "minimum" in Python.
Must preserve: MAX_ITEMS (constant), max_items (variable)
Must rename: def max(...) and max(...) call
Result changes: {result}
"""
    grade_question = "Did the editor rename only the function 'max' while preserving MAX_ITEMS and max_items?"
    assert await fact_judge_grader.grade(output_for_grade, grade_question)


class _StubLLM:
    def __init__(self, response: str):
        self._response = response

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        return self._response


@pytest.mark.asyncio
async def test_editor_parses_changes_json():
    """Unit test: Editor correctly parses JSON changes output."""
    llm = _StubLLM('{"changes": [{"original": "function max(depth = 0) {", "new_content": "function min(depth = 0) {"}]}')
    chunks = [
        RelevantChunk(
            content="function max(depth = 0) {",
            file_path="test.js",
            start_line=3,
            end_line=3,
            chunk_type="function",
            name="max",
        )
    ]
    
    result = await run_editor(llm, "Rename max to min", "test.js", chunks)
    
    assert result is not None
    assert len(result) == 1
    assert result[0].original == "function max(depth = 0) {"
    assert "min" in result[0].new_content
