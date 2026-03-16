import json
import re
from dataclasses import dataclass
from typing import List, Optional
from langsmith import traceable
from .utils import load_prompt
from coding_agent.llm import LLMBackend
from coding_agent.limits import AGENT_MAX_TOKENS
from coding_agent.vectorstore import RelevantChunk


@dataclass
class EditorChange:
    original: str
    new_content: str


def format_chunks_for_prompt(chunks: List[RelevantChunk]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[{chunk.chunk_type}]"
        if chunk.name:
            header += f" {chunk.name}"
        parts.append(f"{header}\n```\n{chunk.content}\n```")
    return "\n\n".join(parts)


def extract_changes_json(response: str) -> Optional[dict]:
    response = response.replace("```json", "").replace("```", "")
    
    match = re.search(r'\{[^{}]*"changes"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    start = response.rfind('{"changes"')
    if start == -1:
        start = response.rfind('{')
    if start == -1:
        return None
    
    brace_count = 0
    for i in range(start, len(response)):
        if response[i] == '{':
            brace_count += 1
        elif response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(response[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


@traceable(run_type="chain", name="editor")
async def run_editor(
    llm_client: LLMBackend,
    task: str,
    file_path: str,
    chunks: List[RelevantChunk],
    error_context: str = "",
    max_tokens: int = AGENT_MAX_TOKENS,
) -> Optional[List[EditorChange]]:
    prompt_template = load_prompt("editor")
    error_section = f"Previous error: {error_context}\n" if error_context else ""
    chunks_text = format_chunks_for_prompt(chunks)
    
    prompt = (prompt_template
        .replace("{task}", task)
        .replace("{file_path}", file_path)
        .replace("{error_context}", error_section)
        .replace("{chunks}", chunks_text))
    
    response = await llm_client.acomplete(prompt, "editor", max_tokens=max_tokens)
    
    result = extract_changes_json(response)
    if not result or "changes" not in result:
        return None
    
    changes = []
    for c in result["changes"]:
        if "original" in c and "new_content" in c:
            changes.append(EditorChange(
                original=c["original"],
                new_content=c["new_content"],
            ))
    
    return changes if changes else None


@traceable(run_type="chain", name="editor_legacy")
async def run_editor_legacy(
    llm_client: LLMBackend,
    task: str,
    file_path: str,
    file_content: str,
    error_context: str = "",
    max_tokens: int = AGENT_MAX_TOKENS,
) -> str | None:
    prompt_template = """You are Editor, a code modification agent. You receive a file and a task, then produce the modified content.

CRITICAL RULES:
1. Output ONLY valid code
2. Preserve all imports, exports, and structure
3. Only change what is necessary for the task
4. Keep indentation and formatting consistent
5. You MUST always wrap the complete modified file between markers:
   - A single line with ---FILE_START---
   - A single line with ---FILE_END---

Response format - output the complete modified file between markers (no explanation):
---FILE_START---
<complete file content>
---FILE_END---

---

Now process:
Task: {task}
File: {file_path}
{error_context}Content:
{file_content}

"""
    error_section = f"Previous error: {error_context}\n" if error_context else ""
    prompt = (prompt_template
        .replace("{task}", task)
        .replace("{file_path}", file_path)
        .replace("{error_context}", error_section)
        .replace("{file_content}", file_content))
    
    response = await llm_client.acomplete(prompt, "editor", max_tokens=max_tokens)
    
    start_match = re.search(r'---FILE_START---', response)
    if not start_match:
        return None
    start_idx = start_match.end()
    end_match = re.search(r'---FILE_END---', response[start_idx:])
    if end_match:
        end_idx = start_idx + end_match.start()
        return response[start_idx:end_idx].strip()
    return response[start_idx:].strip()
