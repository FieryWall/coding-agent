import json
from langsmith import traceable
from .utils import load_prompt
from coding_agent.llm import LLMBackend, extract_json
from coding_agent.limits import AGENT_MAX_TOKENS


@traceable(run_type="chain", name="result_writer")
async def run_result_writer(
    llm_client: LLMBackend,
    user_input: str,
    action: str,
    result_payload: dict,
    max_tokens: int = AGENT_MAX_TOKENS,
) -> dict:
    prompt_template = load_prompt("result_writer")
    payload_str = json.dumps(result_payload, indent=2) if result_payload else "{}"
    prompt = (
        prompt_template.replace("{user_input}", user_input)
        .replace("{action}", action)
        .replace("{result_payload}", payload_str)
    )
    response = await llm_client.acomplete(prompt, "result_writer", max_tokens=max_tokens)
    out = extract_json(response)
    return out if isinstance(out, dict) else {"request": user_input, "action": action, "result": result_payload}
