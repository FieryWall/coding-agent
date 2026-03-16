from coding_agent.agents import run_validator, run_syntax_check
import pytest


@pytest.mark.asyncio
async def test_validator_syntax_error(production_llm_client, fact_judge_grader):
    result = await run_validator(
        production_llm_client,
        "playground/pets/app.js",
        "SyntaxError: Unexpected token '}'",
    )
    assert "fixable" in result
    assert "cause" in result
    assert "suggestion" in result
    assert await fact_judge_grader.grade(str(result), "Is the cause factually relevant to a syntax error about an unexpected token?")


@pytest.mark.asyncio
async def test_validator_pairwise(production_llm_client, nano_llm_client, pairwise_grader):
    prod_result = await run_validator(production_llm_client, "playground/pets/app.js", "SyntaxError: Unexpected token '}'")
    nano_result = await run_validator(nano_llm_client, "playground/pets/app.js", "SyntaxError: Unexpected token '}'")
    score = await pairwise_grader.grade(
        str(prod_result),
        str(nano_result),
        "Relevance of cause and suggestion to the syntax error.",
    )
    assert score >= 0.5


@pytest.mark.asyncio
async def test_validator_invalid_request(production_llm_client):
    result = await run_validator(production_llm_client, "x.js", "")
    assert result["fixable"] is False


@pytest.mark.asyncio
async def test_syntax_check_valid_js(production_llm_client):
    content = "function foo() { return 1; }"
    result = await run_syntax_check(production_llm_client, "playground/pets/app.js", content)
    assert isinstance(result, dict)
    assert result.get("valid", False) is True


@pytest.mark.asyncio
async def test_syntax_check_invalid_js(production_llm_client):
    content = "function foo( { return 1; }"
    result = await run_syntax_check(production_llm_client, "playground/pets/app.js", content)
    assert isinstance(result, dict)
    assert result.get("valid", True) is False
    assert "error" in result
