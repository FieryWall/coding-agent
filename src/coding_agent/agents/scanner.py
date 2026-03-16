import asyncio
import json
from langsmith import traceable
from .utils import load_scanner_prompt
from coding_agent.llm import LLMBackend, extract_json
from coding_agent.tools import ToolsRepository
from coding_agent.limits import AGENT_MAX_TOKENS

MAX_TOOL_CALLS = 3


@traceable(run_type="chain", name="scanner")
async def run_scanner(
    llm_client: LLMBackend,
    tools_repo: ToolsRepository,
    user_input: str,
    project_path: str,
    max_tokens: int = AGENT_MAX_TOKENS,
) -> dict:
    context = ""
    tool_calls = 0
    while tool_calls < MAX_TOOL_CALLS:
        step = "synthesize" if context else "gather"
        prompt = load_scanner_prompt(step, user_input, project_path, context)
        response = await llm_client.acomplete(prompt, "scanner", max_tokens=max_tokens)
        
        if not response or not response.strip():
            print("[WARN] Empty response from LLM")
            break
        
        tool_json = extract_nested_json(response, '"tool"')
        if tool_json and "tool" in tool_json:
            tool_name = tool_json["tool"]
            tool_args = tool_json.get("args", {})
            
            print(f"\n[TOOL] {tool_name}({tool_args})")
            result = await asyncio.to_thread(tools_repo.execute, tool_name, tool_args)
            print(f"[RESULT] {result}")
            
            context = context + f"\n[TOOL] {tool_name} result: {result}\n"
            tool_calls += 1
            continue

        all_json = extract_json(response)
        if all_json and "tool" in all_json:
            tool_calls += 1
            continue

        action_json = extract_action_json(response)
        if action_json and "action" in action_json:
            return action_json

        if all_json and "action" in all_json:
            return all_json
        
        print(f"[DEBUG] Could not parse response: {response[:200]}...")
        break
    
    return {"error": "Scanner failed to produce valid result"}


def extract_action_json(text: str) -> dict | None:
    cleaned = text
    for token in ["<|message|>", "<|end|>", "<|start|>", "```json", "```"]:
        cleaned = cleaned.replace(token, " ")
    for pattern in ('{"action"', '{ "action"'):
        idx = cleaned.find(pattern)
        if idx != -1:
            start = idx
            brace_count = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    brace_count += 1
                elif cleaned[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(cleaned[start:i+1])
                        except json.JSONDecodeError:
                            break
            break
    idx = cleaned.rfind('"action"')
    if idx == -1:
        return None
    start = cleaned.rfind("{", 0, idx + 1)
    if start == -1:
        start = cleaned.find("{", idx)
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            brace_count += 1
        elif cleaned[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(cleaned[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_nested_json(text: str, start_marker: str) -> dict | None:
    cleaned = text
    for token in ["<|message|>", "<|end|>", "<|start|>", "```json", "```"]:
        cleaned = cleaned.replace(token, " ")
    idx = cleaned.rfind(start_marker)
    if idx == -1:
        return None
    start = cleaned.rfind("{", 0, idx + 1)
    if start == -1:
        start = cleaned.find("{", idx)
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            brace_count += 1
        elif cleaned[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(cleaned[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def clean_response(text: str) -> str:
    result = text
    for token in ["<|end|>", "<|start|>", "<|message|>", "<|channel|>"]:
        result = result.split(token)[0] if token in result else result
    return result.strip()


async def run_scanner_step(
    llm_client: LLMBackend,
    step: str,
    user_input: str,
    project_path: str,
    context: str = "",
    max_tokens: int = AGENT_MAX_TOKENS,
) -> str:
    prompt = load_scanner_prompt(step, user_input, project_path, context)
    return await llm_client.acomplete(prompt, f"scanner_step_{step}", max_tokens=max_tokens)