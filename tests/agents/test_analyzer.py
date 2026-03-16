from coding_agent.agents import run_output_analyzer
import pytest


@pytest.mark.asyncio
async def test_analyzer_ok_output(production_llm_client, fact_judge_grader):
    result = await run_output_analyzer(
        production_llm_client,
        "playground/pets/app.js",
        "Hello world",
        'Rename "foo" to "bar"',
    )
    assert "looks_ok" in result
    assert "issue" in result
    assert "suggestion" in result
    output_for_grade = f"Output was 'Hello world'. Task was 'Rename foo to bar'. Analyzer result: {result}"
    assert await fact_judge_grader.grade(output_for_grade, "Does the analyzer's verdict (looks_ok, issue) logically follow from the output and task? Answer YES if the verdict is a reasonable assessment.")


@pytest.mark.asyncio
async def test_analyzer_pairwise(production_llm_client, nano_llm_client, pairwise_grader):
    prod_result = await run_output_analyzer(production_llm_client, "playground/pets/app.js", "Hello world", 'Rename "foo" to "bar"')
    nano_result = await run_output_analyzer(nano_llm_client, "playground/pets/app.js", "Hello world", 'Rename "foo" to "bar"')
    score = await pairwise_grader.grade(
        str(prod_result),
        str(nano_result),
        "Consistency of verdict (looks_ok, issue) with output and task.",
    )
    assert score >= 0.5


@pytest.mark.asyncio
async def test_analyzer_invalid_request(production_llm_client):
    result = await run_output_analyzer(production_llm_client, "x.js", "", "")
    assert "looks_ok" in result and "issue" in result and "suggestion" in result
