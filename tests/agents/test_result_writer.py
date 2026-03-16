from coding_agent.agents import run_result_writer
import pytest


@pytest.mark.asyncio
async def test_result_writer_search_output(production_llm_client, fact_judge_grader):
    payload = {"matches": [{"file": "app.js", "line": 1, "content": "foo()"}]}
    result = await run_result_writer(
        production_llm_client,
        "Find usages of foo and save to JSON",
        "search",
        payload,
    )
    assert isinstance(result, dict)
    assert len(result) >= 1
    output_for_grade = f"User asked: Find usages of foo and save to JSON. Action: search. Raw data had matches. Result JSON: {result}"
    assert await fact_judge_grader.grade(output_for_grade, "Does the JSON sensibly represent a search result (e.g. contains query/request, matches or similar)? Answer YES if it is a reasonable result structure.")


@pytest.mark.asyncio
async def test_result_writer_pairwise(production_llm_client, nano_llm_client, pairwise_grader):
    payload = {"matches": [], "total": 0}
    prod_result = await run_result_writer(production_llm_client, "find bar", "search", payload)
    nano_result = await run_result_writer(nano_llm_client, "find bar", "search", payload)
    score = await pairwise_grader.grade(
        str(prod_result),
        str(nano_result),
        "Consistency of JSON structure and content for the same request.",
    )
    assert score >= 0.5


@pytest.mark.asyncio
async def test_result_writer_invalid_request(production_llm_client):
    result = await run_result_writer(production_llm_client, "", "search", {})
    assert isinstance(result, dict)
    assert len(result) >= 1


class _MockLLM:
    def __init__(self, response: str):
        self._response = response

    async def acomplete(self, prompt, agent_name=None, max_tokens=4096):
        return self._response


@pytest.mark.asyncio
async def test_result_writer_fallback_when_no_json():
    mock_llm = _MockLLM("no json here")
    got = await run_result_writer(mock_llm, "find foo", "search", {"matches": []})
    assert got == {"request": "find foo", "action": "search", "result": {"matches": []}}


@pytest.mark.asyncio
async def test_result_writer_uses_llm_json_when_valid():
    mock_llm = _MockLLM('{"query": "find foo", "count": 0, "items": []}')
    got = await run_result_writer(mock_llm, "find foo", "search", {"matches": []})
    assert got.get("query") == "find foo"
    assert got.get("count") == 0
    assert got.get("items") == []
