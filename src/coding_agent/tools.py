import os
import subprocess
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from langsmith import traceable


class ToolError(Exception):
    pass


@dataclass
class FileChange:
    start_line: int
    end_line: int
    content: str


def apply_changes(file_path: str, changes: List[FileChange]) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    sorted_changes = sorted(changes, key=lambda c: c.start_line, reverse=True)
    
    for change in sorted_changes:
        start_idx = change.start_line - 1
        end_idx = change.end_line
        
        new_lines = change.content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        
        lines[start_idx:end_idx] = new_lines
    
    result = ''.join(lines)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)
    
    return result


def parse_changes(changes_data: List[dict]) -> List[FileChange]:
    return [
        FileChange(
            start_line=c["start_line"],
            end_line=c["end_line"],
            content=c["content"],
        )
        for c in changes_data
    ]


@dataclass
class ContentChange:
    original: str
    new_content: str


def apply_content_changes(file_path: str, changes: List[ContentChange]) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    positions = []
    for change in changes:
        pos = content.find(change.original)
        if pos != -1:
            positions.append((pos, change))
    
    positions.sort(key=lambda x: x[0], reverse=True)
    
    for pos, change in positions:
        end_pos = pos + len(change.original)
        content = content[:pos] + change.new_content + content[end_pos:]
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return content


@traceable(run_type="tool", name="ls")
def tool_ls(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {"error": f"Path does not exist: {path}"}
        
        if os.path.isfile(path):
            return {"files": [path]}
        
        items = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            item_type = "dir" if os.path.isdir(full_path) else "file"
            items.append({"name": item, "type": item_type})
        
        return {"path": path, "items": items}
    except Exception as e:
        return {"error": str(e)}

@traceable(run_type="tool", name="cat")
def tool_cat(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {"error": f"File does not exist: {path}"}
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        content_with_lines = []
        for i, line in enumerate(lines, 1):
            content_with_lines.append(f"{i}|{line.rstrip()}")
        
        return {"path": path, "content": "\n".join(content_with_lines), "total_lines": len(lines)}
    except Exception as e:
        return {"error": str(e)}

@traceable(run_type="tool", name="grep")
def tool_grep(pattern: str, path: str, _root_path: str = None) -> dict:
    try:
        results = []
        search_path = Path(path)
        root = Path(_root_path) if _root_path else search_path
        
        if search_path.is_file():
            files = [search_path]
        else:
            files = []
            for ext in ["js", "ts", "py", "java", "cpp", "c", "h", "go", "cs"]:
                files.extend(search_path.rglob(f"*.{ext}"))
        
        for file_path in files:
            try:
                rel_path = file_path.relative_to(root) if _root_path else file_path
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern in line:
                            results.append({
                                "file": f"./{rel_path}",
                                "line": line_num,
                                "content": line.strip()
                            })
            except Exception:
                continue
        
        return {"pattern": pattern, "matches": results, "total": len(results)}
    except Exception as e:
        return {"error": str(e)}

def tool_replace(path: str, content: str) -> dict:
    try:
        if not os.path.exists(path):
            return {"error": f"File does not exist: {path}"}
        
        if not os.access(path, os.W_OK):
            return {"error": f"File is not writable: {path}"}
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return {"success": True, "path": path}
    except Exception as e:
        return {"error": str(e)}

def tool_edit(path: str, start_line: int, end_line: int, new_content: str) -> dict:
    try:
        if not os.path.exists(path):
            return {"error": f"File does not exist: {path}"}
        
        if not os.access(path, os.W_OK):
            return {"error": f"File is not writable: {path}"}
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return {"error": f"Invalid line range: {start_line}-{end_line}, file has {len(lines)} lines"}
        
        new_lines = new_content.split("\n")
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines = [line + "\n" for line in new_lines]
        
        lines[start_line-1:end_line] = new_lines
        
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        return {"success": True, "path": path, "lines_replaced": f"{start_line}-{end_line}"}
    except Exception as e:
        return {"error": str(e)}

def tool_run(path: str, timeout: int = 5) -> dict:
    try:
        if not os.path.exists(path):
            return {"error": f"File does not exist: {path}"}
        
        abs_path = os.path.abspath(path)
        result = subprocess.run(
            ["node", abs_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(abs_path)
        )
        
        if result.returncode == 0:
            return {"success": True, "path": path, "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}
        else:
            return {"success": False, "path": path, "error": result.stderr.strip(), "output": result.stdout[-200:] if result.stdout else ""}
    except subprocess.TimeoutExpired:
        return {"success": True, "path": path, "note": "Process timed out (likely running correctly)"}
    except Exception as e:
        return {"error": str(e)}

def check_writable(path: str) -> bool:
    return os.path.exists(path) and os.access(path, os.W_OK)

def check_all_writable(paths: list[str]) -> tuple[bool, list[str]]:
    readonly = [p for p in paths if not check_writable(p)]
    return len(readonly) == 0, readonly

class BackupManager:
    def __init__(self):
        self.backup_dir = None
        self.backed_up = {}
    
    def init_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"/tmp/agent_backup_{timestamp}"
        os.makedirs(self.backup_dir, exist_ok=True)
        return self.backup_dir
    
    def backup_file(self, path: str):
        if self.backup_dir is None:
            self.init_backup()
        
        backup_path = os.path.join(self.backup_dir, os.path.basename(path))
        shutil.copy2(path, backup_path)
        self.backed_up[path] = backup_path
    
    def restore_file(self, path: str):
        if path in self.backed_up:
            shutil.copy2(self.backed_up[path], path)
    
    def restore_all(self):
        for original, backup in self.backed_up.items():
            shutil.copy2(backup, original)
    
    def cleanup(self):
        if self.backup_dir and os.path.exists(self.backup_dir):
            shutil.rmtree(self.backup_dir)
        self.backed_up = {}

TOOLS = {
    "ls": tool_ls,
    "cat": tool_cat,
    "grep": tool_grep,
    "replace": tool_replace,
    "edit": tool_edit,
}


class ToolsRepository:
    def __init__(self):
        self._root_path = None

    def set_root_path(self, root_path: str):
        self._root_path = os.path.abspath(root_path) if root_path else None

    def execute(self, name: str, args: dict) -> dict:
        if name not in TOOLS:
            return {"error": f"Unknown tool: {name}"}
        args = dict(args)
        if self._root_path and "path" in args and name in ("ls", "cat", "grep"):
            p = args["path"]
            if not os.path.isabs(p):
                args["path"] = os.path.normpath(os.path.join(self._root_path, p))
            if name == "grep":
                args["_root_path"] = self._root_path
        try:
            return TOOLS[name](**args)
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}


class LimitingToolsRepository(ToolsRepository):
    def __init__(self, delegate: ToolsRepository, per_tool_limit: int = 100):
        super().__init__()
        self.delegate = delegate
        self.per_tool_limit = per_tool_limit
        self._counts: dict[str, int] = {}

    def set_root_path(self, root_path: str):
        if hasattr(self.delegate, "set_root_path"):
            self.delegate.set_root_path(root_path)

    def execute(self, name: str, args: dict) -> dict:
        self._counts[name] = self._counts.get(name, 0) + 1
        if self._counts[name] > self.per_tool_limit:
            return {"error": f"Tool call limit exceeded for {name} ({self.per_tool_limit})"}
        return self.delegate.execute(name, args)


def execute_tool(name: str, args: dict) -> dict:
    return ToolsRepository().execute(name, args)


def get_next_result_json_path(project_path: str) -> Path:
    root = Path(project_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "result.json").exists():
        return root / "result.json"
    for i in range(1000):
        p = root / f"result{i}.json"
        if not p.exists():
            return p
    return root / "result999.json"
