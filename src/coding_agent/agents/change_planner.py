import json
import re
from dataclasses import dataclass
from typing import List, Optional
from langsmith import traceable
from .utils import load_prompt
from coding_agent.llm import LLMBackend
from coding_agent.limits import AGENT_MAX_TOKENS


@dataclass
class PlannedChange:
    original: str
    instruction: str
    skip: bool = False


def extract_changes_json(response: str) -> Optional[dict]:
    response = response.replace("```json", "").replace("```", "")
    
    start = response.rfind('{"changes"')
    if start == -1:
        start = response.rfind('{')
    if start == -1:
        return None
    
    for end in range(len(response) - 1, start, -1):
        if response[end] == '}':
            try:
                result = json.loads(response[start:end+1])
                if isinstance(result, dict) and "changes" in result:
                    return result
            except json.JSONDecodeError:
                continue
    return None


@traceable(run_type="chain", name="change_planner")
async def run_change_planner(
    llm_client: LLMBackend,
    task: str,
    file_path: str,
    file_content: str,
    target: str,
    max_tokens: int = AGENT_MAX_TOKENS,
) -> Optional[List[PlannedChange]]:
    prompt_template = load_prompt("change_planner")
    
    prompt = (prompt_template
        .replace("{task}", task)
        .replace("{file_path}", file_path)
        .replace("{file_content}", file_content)
        .replace("{target}", target))
    
    response = await llm_client.acomplete(prompt, "change_planner", max_tokens=max_tokens)
    
    result = extract_changes_json(response)
    if not result or "changes" not in result:
        return None
    
    changes = []
    for c in result["changes"]:
        if "original" in c and "instruction" in c:
            changes.append(PlannedChange(
                original=c["original"],
                instruction=c["instruction"],
                skip=c.get("skip", False),
            ))
    
    return changes if changes else None
