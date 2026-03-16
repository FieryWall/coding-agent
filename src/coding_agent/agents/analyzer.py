from langsmith import traceable
from .utils import load_prompt
from coding_agent.llm import LLMBackend, extract_json
from coding_agent.limits import AGENT_MAX_TOKENS


@traceable(run_type="chain", name="output_analyzer")
async def run_output_analyzer(
    llm_client: LLMBackend,
    file_path: str,
    output: str,
    task: str,
    max_tokens: int = AGENT_MAX_TOKENS,
) -> dict:
    prompt_template = load_prompt("analyzer")
    prompt = (prompt_template
        .replace("{task}", task)
        .replace("{file_path}", file_path)
        .replace("{output}", output[:2000]))
    response = await llm_client.acomplete(prompt, "output_analyzer", max_tokens=max_tokens)
    result = extract_json(response)
    return result or {"looks_ok": True, "issue": "", "suggestion": ""}
