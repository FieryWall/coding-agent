# Plan: Add LangSmith Observability

## Context

The project already has a `.env` file with all LangSmith credentials set (`LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_API_KEY`) but the SDK is not installed and no code is instrumented. The goal is to wire in `@traceable` decorators and `wrap_openai()` so every user request produces a full, nested trace visible in the LangSmith UI.

---

## Critical Methods to Observe (Priority Order)

| Priority | Method | File | Why |
|---|---|---|---|
| 1 | `process_write_action()` | `main.py` | Top-level write pipeline — shows the full rename/fix flow |
| 2 | `run_scanner()` | `agents/scanner.py` | Intent understanding + tool loop |
| 3 | `run_editor()` | `agents/editor.py` | LLM code modification |
| 4 | `run_syntax_check()` | `agents/validator.py` | Pre/post validation |
| 5 | `run_validator()` | `agents/validator.py` | Retry hint generation |
| 6 | `run_result_writer()` | `agents/result_writer.py` | JSON output generation |
| 7 | `run_output_analyzer()` | `agents/analyzer.py` | Output analysis |
| 8 | `tool_ls / tool_cat / tool_grep` | `tools.py` | Individual tool calls under scanner |

OpenAI API calls are captured automatically by `wrap_openai()` — no manual decoration needed for LLM spans.

---

## Implementation Plan

### Step 1 — Add dependencies

**`pyproject.toml`** — add to `[project] dependencies`:
```
"langsmith",
"python-dotenv",
```

**`requirements.txt`** — append:
```
langsmith
python-dotenv
```

---

### Step 2 — Load `.env` in entry point

**`src/coding_agent/main.py`** — at the top of `main()` (before `asyncio.run(amain())`):
```python
from dotenv import load_dotenv
load_dotenv()
```

This ensures the LangSmith env vars are active for the whole run.

---

### Step 3 — Wrap OpenAI clients

**`src/coding_agent/llm.py`** — two changes:

In `OpenAIBackend.load()` (currently line 117):
```python
# Before:
self.client = OpenAI()

# After:
from langsmith.wrappers import wrap_openai
self.client = wrap_openai(OpenAI())
```

In `OpenAIBackend.agenerate()` (currently line 175):
```python
# Before:
client = AsyncOpenAI()

# After:
from langsmith.wrappers import wrap_openai
client = wrap_openai(AsyncOpenAI())
```

This captures every `chat.completions.create()` call automatically with model, messages, tokens, latency in LangSmith — no other changes needed to llm.py.

---

### Step 4 — Decorate tool functions

**`src/coding_agent/tools.py`** — import and decorate the three read tools used by the scanner:

```python
from langsmith import traceable

@traceable(run_type="tool", name="ls")
def tool_ls(path: str) -> dict:
    ...

@traceable(run_type="tool", name="cat")
def tool_cat(path: str) -> dict:
    ...

@traceable(run_type="tool", name="grep")
def tool_grep(pattern: str, path: str) -> dict:
    ...
```

Write tools (`tool_replace`, `tool_edit`) are called directly from main.py via `open()`, not through the tool dispatch, so they don't need decoration for now.

---

### Step 5 — Decorate agent functions

Each agent file gets one import and one decorator:

**`src/coding_agent/agents/scanner.py`**:
```python
from langsmith import traceable

@traceable(run_type="chain", name="scanner")
async def run_scanner(...) -> dict:
```

**`src/coding_agent/agents/editor.py`**:
```python
from langsmith import traceable

@traceable(run_type="chain", name="editor")
async def run_editor(...) -> str | None:
```

**`src/coding_agent/agents/validator.py`**:
```python
from langsmith import traceable

@traceable(run_type="chain", name="syntax_check")
async def run_syntax_check(...) -> dict:

@traceable(run_type="chain", name="validator")
async def run_validator(...) -> dict:
```

**`src/coding_agent/agents/result_writer.py`**:
```python
from langsmith import traceable

@traceable(run_type="chain", name="result_writer")
async def run_result_writer(...) -> dict:
```

**`src/coding_agent/agents/analyzer.py`**:
```python
from langsmith import traceable

@traceable(run_type="chain", name="output_analyzer")
async def run_output_analyzer(...) -> dict:
```

---

### Step 6 — Decorate the write pipeline

**`src/coding_agent/main.py`**:
```python
from langsmith import traceable

@traceable(run_type="chain", name="write_action")
async def process_write_action(llm_client: LLMBackend, scan_result: dict) -> dict:
    ...
```

This gives LangSmith a parent span that contains every editor, syntax_check, and validator child span within one rename/fix operation.

---

## Files Modified

| File | Change |
|---|---|
| `pyproject.toml` | Add `langsmith`, `python-dotenv` to deps |
| `requirements.txt` | Same |
| `src/coding_agent/main.py` | `load_dotenv()` + `@traceable` on `process_write_action()` |
| `src/coding_agent/llm.py` | `wrap_openai()` on sync and async OpenAI clients |
| `src/coding_agent/tools.py` | `@traceable` on `tool_ls`, `tool_cat`, `tool_grep` |
| `src/coding_agent/agents/scanner.py` | `@traceable` on `run_scanner()` |
| `src/coding_agent/agents/editor.py` | `@traceable` on `run_editor()` |
| `src/coding_agent/agents/validator.py` | `@traceable` on `run_syntax_check()` + `run_validator()` |
| `src/coding_agent/agents/result_writer.py` | `@traceable` on `run_result_writer()` |
| `src/coding_agent/agents/analyzer.py` | `@traceable` on `run_output_analyzer()` |

---

## Expected Trace Structure in LangSmith

Confirmed from live run log (`Rename printStack to train`):
- Scanner made **1 tool call** (grep only, no ls) then returned action JSON
- `syntax_check` was called **3 times**: pre-check, post-edit per file, final check
- `editor` was called once (1st attempt succeeded)
- No `validator` call (no failure in this run)

```
write_action
├── scanner (chain)
│   ├── grep (tool)                         ← scanner used grep directly (skipped ls)
│   └── OpenAI ChatCompletion (auto-captured by wrap_openai)
├── syntax_check (chain) — pre-check
│   └── OpenAI ChatCompletion
├── editor (chain)                          ← per file, up to 3 retries
│   └── OpenAI ChatCompletion
├── syntax_check (chain) — post-edit
│   └── OpenAI ChatCompletion
├── validator (chain) — only on failure
│   └── OpenAI ChatCompletion
├── syntax_check (chain) — final validation
│   └── OpenAI ChatCompletion
└── result_writer (chain) — if output_json=true
    └── OpenAI ChatCompletion
```

Note: The `[agent_name]` prefixes already printed to console (`[scanner]`, `[editor]`, `[syntax_check]`) match the `name=` values in `@traceable` — adding tracing does not change console output.

---

## Verification

1. Run `uv sync` to install new deps
2. Run `uv run coding-agent -p playground`
3. Enter: `Rename function max to maximum`
4. Open https://smith.langchain.com → project `agent-refactorer`
5. Confirm trace tree matches expected structure above with correct inputs/outputs at each node
