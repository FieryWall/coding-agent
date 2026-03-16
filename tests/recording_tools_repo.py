from typing import Callable
from coding_agent.tools import ToolsRepository


class RecordingToolsRepository(ToolsRepository):
    def __init__(self, delegate: ToolsRepository):
        super().__init__()
        self.delegate = delegate
        self.calls: list[dict] = []

    def set_root_path(self, root_path: str):
        if hasattr(self.delegate, "set_root_path"):
            self.delegate.set_root_path(root_path)

    def execute(self, name: str, args: dict) -> dict:
        self.calls.append({"tool": name, "args": dict(args), "order": len(self.calls)})
        return self.delegate.execute(name, args)


def assert_hard_constraints(calls: list[dict], *, must_have_tool: str = None, tool_args_predicate: Callable[[dict], bool] = None):
    if must_have_tool:
        tool_calls = [c for c in calls if c["tool"] == must_have_tool]
        assert len(tool_calls) >= 1, f"Hard: expected at least one call to {must_have_tool}, got {[c['tool'] for c in calls]}"
        if tool_args_predicate:
            assert any(tool_args_predicate(c["args"]) for c in tool_calls), f"Hard: no {must_have_tool} call matched args predicate"


def soft_constraint_ls_before_grep(calls: list[dict]) -> bool:
    indices = {c["tool"]: c["order"] for c in calls}
    if "ls" not in indices or "grep" not in indices:
        return True
    return indices["ls"] < indices["grep"]


def soft_constraint_exact_pattern(calls: list[dict], expected_pattern: str) -> bool:
    for c in calls:
        if c["tool"] == "grep" and c["args"].get("pattern") == expected_pattern:
            return True
    return False


def format_execution_trace(calls: list[dict]) -> str:
    lines = [f"{i + 1}. {c['tool']}({c['args']})" for i, c in enumerate(calls)]
    return "\n".join(lines) if lines else "(no tool calls)"


def assert_edit_tool_called(calls: list[dict], path_predicate=None, min_calls=1):
    edit_calls = [c for c in calls if c["tool"] == "edit"]
    assert len(edit_calls) >= min_calls, f"Expected at least {min_calls} 'edit' call(s), got {[c['tool'] for c in calls]}"
    if path_predicate is not None:
        assert any(path_predicate(c["args"].get("path", "")) for c in edit_calls), "No 'edit' call matched path predicate"


def optimal_trajectory_score(calls: list[dict], ideal_sequence: list[dict]) -> float:
    if not ideal_sequence:
        return 1.0
    actual = [(c["tool"], c["args"]) for c in calls]
    ideal = [(s["tool"], s.get("args", {})) for s in ideal_sequence]
    matches = 0
    j = 0
    for tool, args in actual:
        if j >= len(ideal):
            break
        want_tool, want_args = ideal[j]
        if tool != want_tool:
            continue
        arg_match = all(str(v) in str(args.get(k, "")) or args.get(k) == v for k, v in want_args.items())
        if arg_match:
            matches += 1
            j += 1
    return matches / len(ideal) if ideal else 1.0
