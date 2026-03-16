from coding_agent.agents import run_editor_legacy
import pytest


@pytest.mark.asyncio
async def test_editor_rename_function(production_llm_client, fact_judge_grader):
    content = "1|function calculateTotal(x) {\n2|    return x;\n3|}\n4|\n5|module.exports = { calculateTotal };"
    result = await run_editor_legacy(
        production_llm_client,
        'Rename function "calculateTotal" to "computeSum"',
        "playground/pets/math.js",
        content,
    )
    assert result is not None
    assert "computeSum" in result
    assert "calculateTotal" not in result
    output_for_grade = f"Task: rename calculateTotal to computeSum. Original: function calculateTotal(x) {{ return x; }} module.exports = {{ calculateTotal }}. Result:\n{result}"
    assert await fact_judge_grader.grade(output_for_grade, "Does the result contain the same logic (return x) and only rename the function name and export, with no other changes?")


@pytest.mark.asyncio
async def test_editor_pairwise(production_llm_client, nano_llm_client, pairwise_grader):
    content = "1|function calculateTotal(x) {\n2|    return x;\n3|}\n5|module.exports = { calculateTotal };"
    task = 'Rename function "calculateTotal" to "computeSum"'
    prod_result = await run_editor_legacy(production_llm_client, task, "playground/pets/math.js", content)
    nano_result = await run_editor_legacy(nano_llm_client, task, "playground/pets/math.js", content)
    score = await pairwise_grader.grade(
        str(prod_result) if prod_result else "",
        str(nano_result) if nano_result else "",
        "Correctness of rename (computeSum, no calculateTotal), preservation of logic.",
    )
    assert score >= 0.5


@pytest.mark.asyncio
async def test_editor_invalid_request(production_llm_client):
    content = "1|function foo() { return 1; }"
    result = await run_editor_legacy(production_llm_client, "", "x.js", content)
    assert result is None or isinstance(result, str)


class _StubLLM:
    def __init__(self, response: str):
        self._response = response

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        return self._response


@pytest.mark.asyncio
async def test_editor_parses_without_file_end():
    llm = _StubLLM(
        "One line summary\n---FILE_START---\nconst x = 1\n",
    )
    result = await run_editor_legacy(llm, "task", "x.js", "1|const x = 0\n")
    assert result == "const x = 1"


@pytest.mark.asyncio
async def test_editor_parses_up_to_file_end_when_present():
    llm = _StubLLM(
        "Summary\n---FILE_START---\nconst x = 1\n---FILE_END---\nTRAILING",
    )
    result = await run_editor_legacy(llm, "task", "x.js", "1|const x = 0\n")
    assert result == "const x = 1"
