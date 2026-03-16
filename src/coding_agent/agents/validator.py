from langsmith import traceable
from .utils import load_prompt
from coding_agent.llm import LLMBackend, extract_json
from coding_agent.limits import AGENT_MAX_TOKENS


@traceable(run_type="chain", name="syntax_check")
async def run_syntax_check(
    llm_client: LLMBackend,
    file_path: str,
    content: str,
    max_tokens: int = 2048,
) -> dict:
    prompt_template = load_prompt("syntax_check")
    prompt = prompt_template.replace("{file_path}", file_path).replace("{content}", content)
    response = await llm_client.acomplete(prompt, "syntax_check", max_tokens=max_tokens)
    result = extract_json(response)
    if result is None:
        return {"valid": True}
    return result if isinstance(result, dict) else {"valid": True}


@traceable(run_type="chain", name="validator")
async def run_validator(
    llm_client: LLMBackend,
    file_path: str,
    error_message: str,
    max_tokens: int = AGENT_MAX_TOKENS,
) -> dict:
    prompt_template = load_prompt("validator")
    prompt = prompt_template.replace("{file_path}", file_path).replace("{error_message}", error_message)
    response = await llm_client.acomplete(prompt, "validator", max_tokens=max_tokens)
    result = extract_json(response)
    return result or {"fixable": False, "cause": "Unknown", "suggestion": ""}
