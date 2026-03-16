# Architecture Review: `coding_agent`

## 1. Overview

`coding_agent` is a Python CLI tool that acts as an AI-powered coding assistant. Given a natural-language request, it understands intent, searches a target project, and applies safe, validated code edits. It is built around a multi-agent pipeline backed by an LLM, with pluggable backends (OpenAI API, local GGUF via llama.cpp, local MLX models).

**Entry point**: `coding-agent` → `src/coding_agent/main.py:main()`
**Python package**: `src/coding_agent/`
**Test target project**: `playground/` (JavaScript pet-function demo files)

---

## 2. High-Level Architecture

```
User Input (CLI)
     │
     ▼
┌─────────────────────────────────────────────┐
│                  main.py                    │
│  (REPL loop, orchestration, write control)  │
└──────────┬──────────────────────────────────┘
           │ asyncio pipeline
    ┌──────┴────────────────────────────────────────┐
    │               Agent Pipeline                  │
    │                                               │
    │  Scanner ──tools──► (ls / cat / grep)         │
    │     │                                         │
    │     ▼                                         │
    │  Route: read-only OR write                    │
    │     │                                         │
    │  (write path)                                 │
    │  pre-check permissions + pre-validate syntax  │
    │     │                                         │
    │  BackupManager.backup_file()                  │
    │     │                                         │
    │  Editor ──► write file ──► SyntaxCheck        │
    │     │  (up to 3 retries per file)             │
    │  Validator (on syntax failure, gives hint)    │
    │     │                                         │
    │  Final syntax check on all modified files     │
    │     │                                         │
    │  [optional] ResultWriter ──► result.json      │
    └───────────────────────────────────────────────┘
           │
    ┌──────┴───────────┐
    │   LLM Backend    │
    │ OpenAI / GGUF /  │
    │ MLX (streaming)  │
    └──────────────────┘
```

---

## 3. Module Breakdown

### `src/coding_agent/main.py`
The orchestrator. Implements:
- CLI argument parsing (`--project`, `--model`)
- REPL loop (input → scan → route → write → JSON output)
- `process_write_action()`: permission check → backup → edit loop → rollback on failure
- `llm_validate_files()`: LLM-based pre/post syntax validation for all affected files

Key constants defined here (outside `limits.py`):
```python
MAX_RETRIES = 3      # per-file editor retry attempts
MAX_TOOL_CALLS = 10  # unused in main; scanner has its own local constant
```

### `src/coding_agent/llm.py`
Abstract LLM backend with three concrete implementations.

| Class | Backend | Notes |
|---|---|---|
| `OpenAIBackend` | OpenAI API (`gpt-4o-mini`) | Supports async streaming (`agenerate`) |
| `LlamaCppBackend` | llama.cpp (GGUF) | Sync only; offloaded to thread in `acomplete` |
| `MLXBackend` | mlx-lm | Sync only; offloaded to thread in `acomplete` |

`LLMBackend.complete()` / `acomplete()`:
- Streams tokens, logs to `src/coding_agent/logs/`
- Detects `finish_reason == "length"` → sets `self.truncated = True`
- On truncation, retries with `2× max_tokens` (capped by `OPENAI_MAX_COMPLETION_TOKENS = 16384`)
- Early-stops streaming when JSON brace counting detects a complete object after `<|message|>` sentinel

`create_llm_client()` auto-detects backend from file path suffix (`.gguf` → llama.cpp, directory → MLX).

Helper functions:
- `extract_json(text)` — finds first `{...}` dict in freeform text
- `extract_json_array(text)` — finds first `[...]` list

### `src/coding_agent/limits.py`
Single source of truth for three global limits:

```python
AGENT_TIMEOUT_SEC = 20         # asyncio.wait_for timeout per agent call
AGENT_MAX_TOKENS = 20_000      # default max_tokens passed to LLM
TOOL_CALL_LIMIT_PER_TOOL = 100 # per-tool-name call limit in LimitingToolsRepository
```

### `src/coding_agent/tools.py`
File-system tools exposed to agents + infrastructure helpers.

**Tool functions** (all return `dict` with either data or `{"error": ...}`):

| Tool | Signature | Notes |
|---|---|---|
| `tool_ls` | `(path)` | Lists directory or single file |
| `tool_cat` | `(path)` | Reads file, returns numbered lines (`N\|content`) |
| `tool_grep` | `(pattern, path)` | Pure-Python substring search over common code extensions |
| `tool_replace` | `(path, content)` | Full file overwrite (writable check) |
| `tool_edit` | `(path, start_line, end_line, new_content)` | Line-range replacement |
| `tool_run` | `(path, timeout=5)` | Runs file with `node` (not wired into agent loop currently) |

**`ToolsRepository`**: dispatches tool calls by name; applies `_root_path` prefix for `ls`, `cat`, `grep` (not for `replace`/`edit`).

**`LimitingToolsRepository`** (decorator pattern over `ToolsRepository`): increments per-tool counter; returns `{"error": "Tool call limit exceeded..."}` when limit is hit.

**`BackupManager`**: copies files to `/tmp/agent_backup_<timestamp>/` before edits; `restore_file()` / `restore_all()` / `cleanup()`.

**`get_next_result_json_path()`**: returns `result.json`, then `result0.json`, `result1.json`, … up to 999.

### `src/coding_agent/agents/`
Five async agent functions, each loaded from a text prompt template in `agents/prompts/`.

| Agent | Function | Input | Output |
|---|---|---|---|
| Scanner | `run_scanner()` | user input + project path | `{action, target, new_name, scope, files, data, output_json?}` |
| Editor | `run_editor()` | task + file path + file content + error context | modified file content string |
| SyntaxCheck | `run_syntax_check()` | file path + content | `{valid, error?}` |
| Validator | `run_validator()` | file path + error message | `{fixable, cause, suggestion}` |
| Analyzer | `run_output_analyzer()` | task + file path + program output | `{looks_ok, issue, suggestion}` |
| ResultWriter | `run_result_writer()` | user input + action + payload | JSON dict |

**Scanner** has an internal tool-call loop (up to `MAX_TOOL_CALLS = 3`) that reads tool JSON from LLM response and dispatches via `ToolsRepository`, building up context in the prompt string.

**Editor** uses sentinel markers `---FILE_START---` / `---FILE_END---` to extract the modified file content from the LLM's freeform response.

**Prompts** are stored as `.txt` files with `{placeholder}` substitution. All prompts include worked examples (few-shot).

---

## 4. Data Flow: Write Action

```
user_input
    │
    ▼
run_scanner() → {action: "rename", target: "foo", new_name: "bar", files: [...]}
    │
    ▼
check_all_writable(files)           ← abort if any readonly
    │
    ▼
llm_validate_files() [pre-check]    ← abort if pre-existing syntax errors
    │
    ▼
BackupManager.init_backup()
    │
for each file:
    BackupManager.backup_file()
    tool_cat() → file_content
    for attempt in 1..3:
        run_editor() [with timeout]
        write new content to disk
        run_syntax_check()
        if valid:
            post-cat & verify rename applied
            break
        else:
            BackupManager.restore_file()
            run_validator() [for error hint]
            → retry with error_context
    if all attempts failed:
        failed = True
        break
    │
if failed:
    BackupManager.restore_all()
else:
    llm_validate_files() [post-check]
    if invalid: restore_all()
    else: BackupManager.cleanup()
    │
    ▼
[optional] run_result_writer() → write result.json
```

---

## 5. Testing & Evaluation

### Test Structure
```
tests/
├── conftest.py               # pytest fixtures: LLM clients + graders
├── graders.py                # LLM-as-judge evaluation classes
├── recording_tools_repo.py   # RecordingToolsRepository + trajectory helpers
├── test_result_json.py       # Unit tests: JSON path generation & payload structure
├── test_write_action.py      # Integration tests: process_write_action() with stub LLMs
└── agents/
    ├── test_scanner.py       # Scanner: hard/soft constraints + judge grading
    ├── test_editor.py        # Editor: rename correctness + output parsing
    ├── test_validator.py     # Validator + SyntaxCheck agent tests
    ├── test_result_writer.py # ResultWriter: fallback + LLM JSON tests
    └── test_analyzer.py      # Analyzer: verdict correctness tests
```

### Graders (`tests/graders.py`)

| Grader | Method | Returns | Use |
|---|---|---|---|
| `JudgeGrader` | `grade(output, judgement)` | `float` 0–1 | Rubric/quality scoring |
| `FactJudgeGrader` | `grade(output, question)` | `bool` | YES/NO fact assertion |
| `PairwiseComparisonGrader` | `grade(output_a, output_b, criterion)` | `float` 0–1 | Model A vs B comparison |

All graders call `acomplete()` on a test LLM client (`gpt-4.1-mini`) and parse the JSON response.

### `RecordingToolsRepository`
Wraps a real `ToolsRepository` to record all tool calls with order. Provides:
- `assert_hard_constraints()` — must-have tool, args predicate
- `soft_constraint_ls_before_grep()` — ordering check
- `soft_constraint_exact_pattern()` — grep pattern check
- `optimal_trajectory_score()` — fuzzy match against an ideal sequence
- `format_execution_trace()` — human-readable trace for grader input

### Test mix
- **Pure unit** (`test_result_json.py`, stub-LLM tests in `test_editor.py`, `test_result_writer.py`, `test_write_action.py`): no network calls, fast
- **Integration** (most `agents/` tests): call real OpenAI API — require `OPENAI_API_KEY`

---

## 6. Issues & Observations

### Design / Architecture

1. **`isinstance` check in `acomplete()`** (`llm.py:78`):
   `if isinstance(self, OpenAIBackend)` breaks the Liskov Substitution Principle. The async path should be defined as an abstract method on `LLMBackend` rather than branching by concrete type.

2. **Duplicate `MAX_TOOL_CALLS` constant**:
   `main.py` defines `MAX_TOOL_CALLS = 10` (unused) while `scanner.py` defines its own `MAX_TOOL_CALLS = 3`. The main.py constant is dead code.

3. **Root path not applied to `replace`/`edit` tools** (`tools.py:196`):
   `ToolsRepository.execute()` only prefixes `_root_path` for `ls`, `cat`, `grep`. Write tools (`replace`, `edit`) receive paths as-is, which means the Scanner must return absolute paths for writes to work correctly.

4. **Backup collision on same basename** (`tools.py:159`):
   `BackupManager.backup_file()` uses `os.path.basename(path)` as the backup filename. If two files in different directories share the same name, the second backup silently overwrites the first.

5. **Vestigial `src/coding_agent/agents.py` top-level file**:
   There is both a `src/coding_agent/agents.py` file and a `src/coding_agent/agents/` package. Python will use the package. The standalone file appears to be an old module that was never removed.

6. **Analyzer agent (`run_output_analyzer`) is unused in `main.py`**:
   It is exported from `agents/__init__.py` and has tests, but the main orchestration loop never calls it. It may be intended for a future step.

7. **Scanner tool loop is single-threaded and prompt-appending**:
   The scanner builds context by appending tool calls and results directly into the prompt string. This grows unboundedly with each tool call and can hit token limits for large projects.

### Error Handling

8. **`asyncio.TimeoutError` → `None` → unchecked `None` branch** (`main.py:83-87`):
   When editor times out, `new_content = None` is set, then `error_context = "Editor timed out"`, but the `continue` jumps to the next attempt without checking. The next iteration calls `run_editor()` again with `error_context = "Editor timed out"` which is correct — but this is a subtle flow and could be made clearer.

9. **Syntax check fallback is permissive** (`validator.py:17`):
   If `extract_json()` returns `None` (malformed LLM output), `run_syntax_check()` returns `{"valid": True}`, silently passing invalid files through validation.

10. **Token doubling may exceed model cap silently**:
    `complete()` doubles `max_tokens` on truncation, but `AGENT_MAX_TOKENS = 20_000 * 2 = 40_000` exceeds `OPENAI_MAX_COMPLETION_TOKENS = 16_384`. The `min()` cap is applied inside `generate()`, but the log will show 40,000 tokens even though the actual cap is 16,384, which is confusing.

### Testing

11. **Phantom model names in conftest**:
    `TEST_LLM = "gpt-4.1-mini"` and `NANO_LLM = "gpt-5-nano"` are not real OpenAI model identifiers as of the codebase date and will cause test failures unless these models exist.

12. **Pairwise score threshold is asymmetric**:
    Scanner pairwise test asserts `score >= 0.3` (production vs nano), while editor/validator/result_writer assert `score >= 0.5`. The lower bar for scanner makes that test nearly always pass.

---

## 7. Strengths

- **Clean agent separation**: each agent has a single responsibility and communicates only through its return value; no shared mutable state between agents.
- **Graceful degradation**: every agent has a fallback return value (e.g., `{"valid": True}`, `{"fixable": False, ...}`, `{"request": ..., "action": ..., "result": ...}`).
- **Comprehensive backup/rollback**: pre-edit permission checks, per-file backup, per-attempt rollback, final full rollback on any failure.
- **Evaluation framework**: uses three distinct grading strategies (rubric, fact, pairwise) with a recording proxy for trajectory analysis — well-suited for systematic LLM output evaluation.
- **Multi-backend LLM support**: single abstraction covers cloud API + two local inference stacks (llama.cpp, MLX), useful for offline or cost-sensitive use.
- **Streaming + early-stop**: JSON completion is detected by brace counting during streaming, avoiding unnecessary token generation.

---

## 8. Summary Table

| Concern | Implementation | Quality |
|---|---|---|
| Agent pipeline | Scanner → Editor → Validator → ResultWriter | Good |
| Error recovery | Backup + per-file rollback + full rollback | Good |
| Limits enforcement | Timeout, token cap, tool call counter | Good |
| LLM abstraction | Abstract base + 3 backends | Good (one LSP issue) |
| Prompt design | Few-shot text templates | Good |
| Tool safety | Writable check, root path prefix (partial) | Partial |
| Test coverage | Unit + integration + LLM-as-judge | Good |
| Code organisation | Clean package structure, one dead file | Good |
