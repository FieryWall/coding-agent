import pytest
from coding_agent.agents import run_change_planner
from coding_agent.agents.change_planner import extract_changes_json


JS_CODE_WITH_MULTIPLE_MAX = """const { max } = require('./pets/max');
const MAX_DEPTH = 10;

console.log('max is the best');

function buddy() {
    max();
    return max;
}

module.exports = { max, MAX_DEPTH };
"""


JS_CODE_WITH_ONLY_CONSTANT = """const MAX_VALUE = 100;

function calculate() {
    if (value > MAX_VALUE) return MAX_VALUE;
    return value;
}
"""


PYTHON_CODE_WITH_MIXED = """MAX_ITEMS = 100
max_items = 50

def max(a, b):
    return a if a > b else b

result = max(MAX_ITEMS, max_items)
"""


class _StubLLM:
    def __init__(self, response: str):
        self._response = response

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        return self._response


class TestExtractChangesJson:
    def test_simple_json(self):
        response = '{"changes": [{"original": "max()", "instruction": "rename", "skip": false}]}'
        result = extract_changes_json(response)
        assert result is not None
        assert len(result["changes"]) == 1

    def test_json_with_braces_in_strings(self):
        response = '{"changes": [{"original": "const { max } = require(\'./pets/max\');", "instruction": "rename import", "skip": false}]}'
        result = extract_changes_json(response)
        assert result is not None
        assert len(result["changes"]) == 1
        assert "{ max }" in result["changes"][0]["original"]

    def test_json_with_nested_braces(self):
        response = '{"changes": [{"original": "module.exports = { max, MAX_DEPTH };", "instruction": "rename export", "skip": false}]}'
        result = extract_changes_json(response)
        assert result is not None
        assert "{ max, MAX_DEPTH }" in result["changes"][0]["original"]

    def test_json_with_multiple_braces_in_code(self):
        response = '{"changes": [{"original": "function test() { return { a: 1 }; }", "instruction": "test", "skip": false}]}'
        result = extract_changes_json(response)
        assert result is not None
        assert "{ return { a: 1 }; }" in result["changes"][0]["original"]

    def test_json_with_prefix_text(self):
        response = 'Here is the analysis:\n{"changes": [{"original": "max()", "instruction": "rename", "skip": false}]}'
        result = extract_changes_json(response)
        assert result is not None
        assert len(result["changes"]) == 1

    def test_json_with_markdown_block(self):
        response = '```json\n{"changes": [{"original": "const { max } = require", "instruction": "rename", "skip": false}]}\n```'
        result = extract_changes_json(response)
        assert result is not None
        assert len(result["changes"]) == 1

    def test_invalid_json_returns_none(self):
        response = 'not valid json at all'
        result = extract_changes_json(response)
        assert result is None

    def test_json_without_changes_key_returns_none(self):
        response = '{"data": [{"original": "test"}]}'
        result = extract_changes_json(response)
        assert result is None


@pytest.mark.asyncio
async def test_planner_parses_changes_json():
    response = '{"changes": [{"original": "const { max } = require", "instruction": "rename import", "skip": false}, {"original": "MAX_DEPTH", "instruction": "constant - skip", "skip": true}]}'
    llm = _StubLLM(response)
    
    result = await run_change_planner(llm, "Rename max to min", "./app.js", "const max = 1", "max")
    
    assert result is not None
    assert len(result) == 2
    assert result[0].original == "const { max } = require"
    assert result[0].skip is False
    assert result[1].skip is True


@pytest.mark.asyncio
async def test_planner_identifies_all_occurrences(production_llm_client):
    result = await run_change_planner(
        production_llm_client,
        'Rename function "max" to "min"',
        "./app.js",
        JS_CODE_WITH_MULTIPLE_MAX,
        "max"
    )
    
    assert result is not None
    assert len(result) >= 4, f"Should find at least 4 occurrences of 'max', found {len(result)}"
    
    active = [c for c in result if not c.skip]
    skipped = [c for c in result if c.skip]
    
    assert len(active) >= 3, "Should have at least 3 changes to make (import, call, export)"
    assert len(skipped) >= 1, "Should skip at least 1 (string literal or MAX_DEPTH)"


@pytest.mark.asyncio
async def test_planner_skips_similar_names(production_llm_client, fact_judge_grader):
    result = await run_change_planner(
        production_llm_client,
        'Rename function "max" to "min"',
        "./app.js",
        JS_CODE_WITH_MULTIPLE_MAX,
        "max"
    )
    
    assert result is not None
    
    max_depth_only_changes = [c for c in result if "MAX_DEPTH" in c.original and "max" not in c.original.replace("MAX_DEPTH", "")]
    for c in max_depth_only_changes:
        assert c.skip is True, "Lines with only MAX_DEPTH should be skipped"
    
    string_only_changes = [c for c in result if c.original.strip() == "console.log('max is the best');"]
    for c in string_only_changes:
        assert c.skip is True, "String literals should be marked as skip"
    
    output = f"Task: Find all 'max' occurrences and classify them\nResult: {result}"
    question = "Did the planner correctly identify which occurrences of 'max' to rename vs skip?"
    assert await fact_judge_grader.grade(output, question)


@pytest.mark.asyncio
async def test_planner_all_skipped_when_no_target(production_llm_client):
    result = await run_change_planner(
        production_llm_client,
        'Rename function "max" to "min"',
        "./utils.js",
        JS_CODE_WITH_ONLY_CONSTANT,
        "max"
    )
    
    if result:
        active = [c for c in result if not c.skip]
        assert len(active) == 0, "All changes should be skipped since there's no function 'max'"


@pytest.mark.asyncio
async def test_planner_python_disambiguation(production_llm_client):
    result = await run_change_planner(
        production_llm_client,
        'Rename function "max" to "minimum"',
        "./utils.py",
        PYTHON_CODE_WITH_MIXED,
        "max"
    )
    
    assert result is not None
    assert len(result) >= 2, "Should find at least 2 occurrences"
    
    active = [c for c in result if not c.skip]
    assert len(active) >= 1, "Should have at least one active change"
    
    all_instructions = " ".join(c.instruction for c in result)
    assert "minimum" in all_instructions.lower(), "Instructions should mention the new name"


@pytest.mark.asyncio
async def test_planner_provides_clear_instructions(production_llm_client):
    result = await run_change_planner(
        production_llm_client,
        'Rename function "max" to "min"',
        "./app.js",
        JS_CODE_WITH_MULTIPLE_MAX,
        "max"
    )
    
    assert result is not None
    
    for change in result:
        assert change.instruction, "Each change should have an instruction"
        assert len(change.instruction) > 5, "Instruction should be meaningful"
