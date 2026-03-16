import argparse
import asyncio
import json
import select
import sys
from pathlib import Path

from langsmith import traceable

from coding_agent.agents import (
    run_scanner, run_editor, run_editor_legacy, run_validator, 
    run_syntax_check, run_result_writer, run_change_planner
)
from coding_agent.agents.editor import EditorChange
from coding_agent.vectorstore import RelevantChunk
from coding_agent.llm import create_llm_client, LLMBackend
from coding_agent.tools import (
    tool_cat,
    check_all_writable, BackupManager, ToolsRepository, LimitingToolsRepository,
    get_next_result_json_path, apply_content_changes, ContentChange,
)
from coding_agent.vectorstore import ProjectIndex
from coding_agent import limits

MAX_RETRIES = 3
MAX_TOOL_CALLS = 10


async def llm_validate_files(llm_client: LLMBackend, files: list[str]) -> tuple[bool, list[str]]:
    errors = []
    for f in files:
        cat_result = tool_cat(f)
        if "error" in cat_result:
            errors.append(f"{f}: {cat_result['error']}")
            continue
        result = await run_syntax_check(llm_client, f, cat_result["content"])
        if not result.get("valid", False):
            errors.append(f"{f}: {result.get('error', 'Unknown error')}")
    return len(errors) == 0, errors

def _build_task_description(action: str, target: str, new_name: str, data: dict, file_path: str) -> str:
    if action == "rename":
        task = f'Rename function/variable "{target}" to "{new_name}"'
        task += f'\nRename ONLY identifiers named exactly "{target}" (function declarations, calls, imports, exports)'
        task += f'\nDO NOT rename: constants, strings, comments, or similar names like "{target.upper()}" or "{target.upper()}_*"'
        
        occurrences = data.get("occurrences", [])
        file_occurrences = [o for o in occurrences if o.get("file") == file_path]
        if file_occurrences:
            lines = [str(o.get("line")) for o in file_occurrences if o.get("line")]
            if lines:
                task += f'\nTarget lines in this file: {", ".join(lines)}'
        
        return task
    
    return f"Fix {target}"


def _resolve_path(path: str, root: str) -> str:
    if Path(path).is_absolute():
        return path
    return str(Path(root) / path)


def _to_relative(path: str, root: str) -> str:
    try:
        return "./" + str(Path(path).relative_to(root))
    except ValueError:
        return path


def _format_paths_for_log(scan_result: dict, root: str) -> dict:
    result = dict(scan_result)
    if "files" in result:
        result["files"] = [_to_relative(f, root) for f in result["files"]]
    if "scope" in result and result["scope"]:
        result["scope"] = _to_relative(result["scope"], root)
    if "modified" in result:
        result["modified"] = [_to_relative(f, root) for f in result["modified"]]
    return result


def _resolve_paths_in_result(scan_result: dict, root: str) -> dict:
    if "error" in scan_result:
        return scan_result
    
    result = dict(scan_result)
    
    if "files" in result:
        result["files"] = [_resolve_path(f, root) for f in result["files"]]
    
    if "scope" in result and result["scope"]:
        result["scope"] = _resolve_path(result["scope"], root)
    
    data = result.get("data", {})
    if data:
        data = dict(data)
        if "occurrences" in data:
            data["occurrences"] = [
                {**o, "file": _resolve_path(o["file"], root)} if "file" in o else o
                for o in data["occurrences"]
            ]
        if "matches" in data:
            data["matches"] = [
                {**m, "file": _resolve_path(m["file"], root)} if "file" in m else m
                for m in data["matches"]
            ]
        result["data"] = data
    
    return result


@traceable(run_type="chain", name="write_action")
async def process_write_action(
    llm_client: LLMBackend,
    scan_result: dict,
    project_index: ProjectIndex,
    root_path: str = "",
    agent_timeout: int = limits.AGENT_TIMEOUT_SEC,
) -> dict:
    action = scan_result.get("action")
    target = scan_result.get("target", "")
    new_name = scan_result.get("new_name", "")
    files = scan_result.get("files", [])
    
    if not files:
        return {"success": False, "error": "No files to modify"}
    
    files_rel = [_to_relative(f, root_path) for f in files] if root_path else files
    print(f"\n[INFO] Files to modify: {files_rel}")
    
    all_writable, readonly = check_all_writable(files)
    if not all_writable:
        return {"success": False, "error": f"Readonly files: {readonly}"}
    
    print("[INFO] Checking syntax before modifications...")
    valid, errors = await llm_validate_files(llm_client, files)
    if not valid:
        return {"success": False, "error": f"Syntax errors before edit: {errors}"}
    
    backup = BackupManager()
    backup.init_backup()
    print(f"[INFO] Backup created at {backup.backup_dir}")
    
    if action == "rename":
        base_task = f'Rename "{target}" to "{new_name}"'
    elif action == "implement":
        base_task = scan_result.get("data", {}).get("task", f"Implement {target}")
    else:
        base_task = f"Fix {target}"
    
    modified_files = []
    failed = False
    
    for file_path in files:
        file_rel = _to_relative(file_path, root_path) if root_path else file_path
        print(f"\n[INFO] Processing: {file_rel}")
        backup.backup_file(file_path)
        
        cat_result = tool_cat(file_path)
        if "error" in cat_result:
            print(f"[ERROR] Cannot read {file_rel}: {cat_result['error']}")
            failed = True
            break
        
        file_content = cat_result["content"]

        # For non-rename actions (fix, etc.), use vectorstore chunks when available.
        # This sends only the relevant function/class chunk to the editor, not the entire file.
        if action != "rename":
            chunks = None
            if project_index:
                chunks = project_index.get_chunks_containing(file_path, target, max_chunks=1)

            if chunks:
                print(f"[INFO] Action '{action}': using chunk-based editor for {file_rel} ({len(chunks)} chunk(s))")
                success = await _process_file_with_chunks(
                    llm_client, file_path, chunks, base_task, backup, agent_timeout
                )
            else:
                print(f"[INFO] Action '{action}': no chunks found, using legacy editor for {file_rel}")
                success = await _process_file_legacy(
                    llm_client, file_path, file_content, base_task, action, target, new_name, backup, agent_timeout
                )
            if success:
                modified_files.append(file_path)
            else:
                failed = True
                break
            continue

        print(f"[PLANNER] Planning changes for {file_rel}...")
        try:
            planned_changes = await asyncio.wait_for(
                run_change_planner(llm_client, base_task, file_path, file_content, target),
                timeout=agent_timeout,
            )
        except asyncio.TimeoutError:
            print(f"[ERROR] Planner timeout for {file_rel}")
            planned_changes = None
        except Exception as e:
            print(f"[ERROR] Planner failed: {type(e).__name__}: {e}")
            planned_changes = None

        if not planned_changes:
            print(f"[INFO] Planner failed, using legacy editor for {file_rel}")
            success = await _process_file_legacy(
                llm_client, file_path, file_content, base_task, action, target, new_name, backup, agent_timeout
            )
            if success:
                modified_files.append(file_path)
            else:
                failed = True
                break
            continue
        
        active_changes = [c for c in planned_changes if not c.skip]
        print(f"[PLANNER] Found {len(planned_changes)} occurrences, {len(active_changes)} to modify")
        
        if not active_changes:
            print(f"[INFO] No changes needed for {file_rel}")
            continue
        
        file_success = await _process_file_with_planner(
            llm_client, file_path, active_changes, action, new_name, backup, agent_timeout,
            project_index=project_index,
        )
        
        if file_success:
            modified_files.append(file_path)
        else:
            failed = True
            break
    
    if failed:
        print("\n[ROLLBACK] Restoring all files...")
        backup.restore_all()
        backup.cleanup()
        return {"success": False, "error": "Failed to modify all files, rolled back"}
    
    if not modified_files:
        backup.restore_all()
        backup.cleanup()
        return {"success": False, "error": "No files were modified"}
    
    print("\n[INFO] Final syntax validation...")
    valid, errors = await llm_validate_files(llm_client, files)
    if not valid:
        print("[ROLLBACK] Syntax validation failed, restoring...")
        backup.restore_all()
        backup.cleanup()
        return {"success": False, "error": f"Syntax validation failed: {errors}"}
    
    backup.cleanup()
    return {"success": True, "modified": modified_files}


async def _process_file_with_planner(
    llm_client: LLMBackend,
    file_path: str,
    planned_changes: list,
    action: str,
    new_name: str,
    backup: BackupManager,
    agent_timeout: int = limits.AGENT_TIMEOUT_SEC,
    project_index: ProjectIndex = None,
) -> bool:
    error_context = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[INFO] Attempt {attempt}/{MAX_RETRIES}")
        
        all_content_changes = []
        
        for i, change in enumerate(planned_changes):
            print(f"[EDITOR] Change {i+1}/{len(planned_changes)}: {change.instruction[:50]}...")

            chunks = None
            if project_index:
                chunks = project_index.get_chunks_containing(file_path, change.original, max_chunks=1)
            if not chunks:
                chunks = [RelevantChunk(
                    content=change.original,
                    file_path=file_path,
                    start_line=0,
                    end_line=0,
                    chunk_type="target",
                    name=f"change_{i+1}",
                )]

            try:
                editor_changes = await asyncio.wait_for(
                    run_editor(
                        llm_client,
                        change.instruction,
                        file_path,
                        chunks,
                        error_context
                    ),
                    timeout=agent_timeout,
                )
            except asyncio.TimeoutError:
                print(f"[ERROR] Editor timeout for change {i+1}")
                editor_changes = None
            except Exception as e:
                print(f"[ERROR] Editor failed for change {i+1}: {type(e).__name__}: {e}")
                editor_changes = None
            
            if editor_changes:
                for ec in editor_changes:
                    all_content_changes.append(ContentChange(ec.original, ec.new_content))
        
        if not all_content_changes:
            error_context = "Editor failed to produce valid changes"
            continue
        
        print(f"[INFO] Applying {len(all_content_changes)} changes")
        
        try:
            new_content = apply_content_changes(file_path, all_content_changes)
        except Exception as e:
            error_context = f"Failed to apply changes: {e}"
            backup.restore_file(file_path)
            continue
        
        validation = await run_syntax_check(llm_client, file_path, new_content)
        if validation.get("valid", False):
            if action == "rename" and new_name and new_name not in new_content:
                print(f"[ERROR] Rename failed: '{new_name}' not found in output")
                error_context = f"Rename failed: '{new_name}' not found in output"
                backup.restore_file(file_path)
                continue
            
            print(f"[SUCCESS] {file_path} validated")
            return True
        else:
            error_msg = validation.get("error", "Unknown syntax error")
            print(f"[ERROR] Validation failed: {error_msg}")
            
            backup.restore_file(file_path)
            
            try:
                validator_result = await asyncio.wait_for(
                    run_validator(llm_client, file_path, error_msg),
                    timeout=agent_timeout,
                )
            except asyncio.TimeoutError:
                validator_result = {"suggestion": "Validator timed out"}
            error_context = f"{error_msg}\nAnalysis: {validator_result.get('suggestion', '')}"
            
            if attempt == MAX_RETRIES:
                print(f"[FATAL] Max retries exceeded for {file_path}")
                return False
    
    return False


async def _process_file_with_chunks(
    llm_client: LLMBackend,
    file_path: str,
    chunks: list,
    task: str,
    backup: BackupManager,
    agent_timeout: int = limits.AGENT_TIMEOUT_SEC,
) -> bool:
    """Process a file using vectorstore chunks instead of the full file content."""
    error_context = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[INFO] Chunk-based attempt {attempt}/{MAX_RETRIES}")

        try:
            editor_changes = await asyncio.wait_for(
                run_editor(llm_client, task, file_path, chunks, error_context),
                timeout=agent_timeout,
            )
        except asyncio.TimeoutError:
            editor_changes = None
            error_context = "Editor timed out"
        except Exception as e:
            print(f"[ERROR] Editor failed: {type(e).__name__}: {e}")
            editor_changes = None

        if not editor_changes:
            error_context = error_context or "Editor failed to produce valid changes"
            continue

        content_changes = [ContentChange(ec.original, ec.new_content) for ec in editor_changes]
        print(f"[INFO] Applying {len(content_changes)} changes")

        try:
            new_content = apply_content_changes(file_path, content_changes)
        except Exception as e:
            error_context = f"Failed to apply changes: {e}"
            backup.restore_file(file_path)
            continue

        validation = await run_syntax_check(llm_client, file_path, new_content)
        if validation.get("valid", False):
            print(f"[SUCCESS] {file_path} validated")
            return True
        else:
            error_msg = validation.get("error", "Unknown syntax error")
            print(f"[ERROR] Validation failed: {error_msg}")
            backup.restore_file(file_path)

            try:
                validator_result = await asyncio.wait_for(
                    run_validator(llm_client, file_path, error_msg),
                    timeout=agent_timeout,
                )
            except asyncio.TimeoutError:
                validator_result = {"suggestion": "Validator timed out"}
            error_context = f"{error_msg}\nAnalysis: {validator_result.get('suggestion', '')}"

            if attempt == MAX_RETRIES:
                print(f"[FATAL] Max retries exceeded for {file_path}")
                return False

    return False


async def _process_file_legacy(
    llm_client: LLMBackend,
    file_path: str,
    file_content: str,
    task: str,
    action: str,
    target: str,
    new_name: str,
    backup: BackupManager,
    agent_timeout: int = limits.AGENT_TIMEOUT_SEC,
) -> bool:
    error_context = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[INFO] Legacy attempt {attempt}/{MAX_RETRIES}")
        
        try:
            new_content = await asyncio.wait_for(
                run_editor_legacy(llm_client, task, file_path, file_content, error_context),
                timeout=agent_timeout,
            )
        except asyncio.TimeoutError:
            new_content = None
            error_context = "Editor timed out"
        
        if new_content is None:
            error_context = "Editor failed to produce valid output"
            continue
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        validation = await run_syntax_check(llm_client, file_path, new_content)
        if validation.get("valid", False):
            if action == "rename" and new_name and new_name not in new_content:
                error_context = f"Rename failed: '{new_name}' not found in output"
                backup.restore_file(file_path)
                continue
            return True
        else:
            error_msg = validation.get("error", "Unknown syntax error")
            print(f"[ERROR] Validation failed: {error_msg}")
            backup.restore_file(file_path)
            
            try:
                validator_result = await asyncio.wait_for(
                    run_validator(llm_client, file_path, error_msg),
                    timeout=agent_timeout,
                )
            except asyncio.TimeoutError:
                validator_result = {"suggestion": "Validator timed out"}
            error_context = f"{error_msg}\nAnalysis: {validator_result.get('suggestion', '')}"
    
    return False


@traceable(run_type="chain", name="write_action_legacy")
async def process_write_action_legacy(
    llm_client: LLMBackend, scan_result: dict, root_path: str = "",
    agent_timeout: int = limits.AGENT_TIMEOUT_SEC,
) -> dict:
    action = scan_result.get("action")
    target = scan_result.get("target", "")
    new_name = scan_result.get("new_name", "")
    files = scan_result.get("files", [])
    
    if not files:
        return {"success": False, "error": "No files to modify"}
    
    files_rel = [_to_relative(f, root_path) for f in files] if root_path else files
    print(f"\n[INFO] Files to modify: {files_rel}")
    
    all_writable, readonly = check_all_writable(files)
    if not all_writable:
        return {"success": False, "error": f"Readonly files: {readonly}"}
    
    print("[INFO] Checking syntax before modifications...")
    valid, errors = await llm_validate_files(llm_client, files)
    if not valid:
        return {"success": False, "error": f"Syntax errors before edit: {errors}"}
    
    backup = BackupManager()
    backup.init_backup()
    print(f"[INFO] Backup created at {backup.backup_dir}")
    
    if action == "rename":
        task = f'Rename "{target}" to "{new_name}"'
    elif action == "implement":
        task = scan_result.get("data", {}).get("task", f"Implement {target}")
    else:
        task = f"Fix {target}"
    
    modified_files = []
    failed = False
    
    for file_path in files:
        file_rel = _to_relative(file_path, root_path) if root_path else file_path
        print(f"\n[INFO] Processing: {file_rel}")
        backup.backup_file(file_path)
        
        cat_result = tool_cat(file_path)
        if "error" in cat_result:
            print(f"[ERROR] Cannot read {file_rel}: {cat_result['error']}")
            failed = True
            break
        
        success = await _process_file_legacy(
            llm_client, file_path, cat_result["content"], task, action, target, new_name, backup, agent_timeout
        )
        if success:
            modified_files.append(file_path)
        else:
            failed = True
            break
    
    if failed:
        print("\n[ROLLBACK] Restoring all files...")
        backup.restore_all()
        backup.cleanup()
        return {"success": False, "error": "Failed to modify all files, rolled back"}
    
    if not modified_files:
        backup.restore_all()
        backup.cleanup()
        return {"success": False, "error": "No files were modified"}
    
    print("\n[INFO] Final syntax validation...")
    valid, errors = await llm_validate_files(llm_client, files)
    if not valid:
        print("[ROLLBACK] Syntax validation failed, restoring...")
        backup.restore_all()
        backup.cleanup()
        return {"success": False, "error": f"Syntax validation failed: {errors}"}
    
    backup.cleanup()
    return {"success": True, "modified": modified_files}


async def amain():
    parser = argparse.ArgumentParser(description="Coding Assistant Agent")
    parser.add_argument("--project", "-p", default="playground/pets", help="Project path")
    parser.add_argument("--model", "-m", help="Path to model (GGUF file, MLX directory, or http(s):// URL for remote server)")
    parser.add_argument("--remote-model", default="local", help="Model name to request from remote server (used when --model is a URL)")
    parser.add_argument("--no-index", action="store_true", help="Disable vectorstore indexing")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        print(f"[FATAL] Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)
    if not project_path.is_dir():
        print(f"[FATAL] Project path is not a directory: {project_path}", file=sys.stderr)
        sys.exit(1)
    args.project = str(project_path)

    print("=" * 60)
    print("Coding Assistant Agent")
    print(f"Project: {args.project}")
    print("=" * 60)

    llm_client = create_llm_client(args.model, remote_model=args.remote_model)
    tools_repo = LimitingToolsRepository(ToolsRepository(), per_tool_limit=limits.TOOL_CALL_LIMIT_PER_TOOL)
    tools_repo.set_root_path(args.project)
    use_local = args.model is not None
    agent_timeout = limits.AGENT_TIMEOUT_SEC_LOCAL if use_local else limits.AGENT_TIMEOUT_SEC
    scanner_timeout = limits.SCANNER_TIMEOUT_SEC_LOCAL if use_local else limits.SCANNER_TIMEOUT_SEC
    
    project_index = None
    if not args.no_index:
        print("\n[INDEX] Indexing project...")
        project_index = ProjectIndex()
        chunk_count = project_index.index_project(args.project)
        print(f"[INDEX] Indexed {chunk_count} code chunks")

    try:
        while True:
            try:
                print("\nYou: ", end="", flush=True)
                first_line = sys.stdin.readline()
                if not first_line:
                    break  # EOF
                lines = [first_line]
                while select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    lines.append(line)
                user_input = "".join(lines).strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                print("\n[SCANNER] Analyzing request...")
                try:
                    scan_result = await asyncio.wait_for(
                        run_scanner(llm_client, tools_repo, user_input, "."),
                        timeout=scanner_timeout,
                    )
                except asyncio.TimeoutError:
                    scan_result = {"error": "Scanner timed out"}
                
                scan_result = _resolve_paths_in_result(scan_result, args.project)

                if "error" in scan_result:
                    print(f"\n[ERROR] {scan_result['error']}")
                    continue

                action = scan_result.get("action")
                print(f"\n[INFO] Action: {action}")
                print(f"[INFO] Scan result: {_format_paths_for_log(scan_result, args.project)}")

                write_result = None
                if action in ("search", "explain"):
                    print("\n[RESULT]")
                    if action == "search":
                        for match in scan_result.get("data", {}).get("matches", []):
                            print(f"  {match['file']}:{match['line']} - {match['content']}")
                    else:
                        print(f"  {scan_result.get('data', {}).get('definition', 'No definition found')}")

                elif action in ("rename", "fix", "implement"):
                    if action == "implement":
                        scan_result.setdefault("data", {})["task"] = user_input
                    print("\n[INFO] Starting write operation...")
                    if project_index:
                        write_result = await process_write_action(llm_client, scan_result, project_index, args.project, agent_timeout)
                    else:
                        write_result = await process_write_action_legacy(llm_client, scan_result, args.project, agent_timeout)

                    if write_result.get("success"):
                        modified_rel = [_to_relative(f, args.project) for f in write_result.get('modified', [])]
                        print(f"\n[SUCCESS] Modified files: {modified_rel}")
                        if project_index:
                            print("[INDEX] Re-indexing modified files...")
                            project_index.index_project(args.project)
                    else:
                        print(f"\n[FAILED] {write_result.get('error', 'Unknown error')}")

                else:
                    print(f"\n[WARN] Unknown action: {action}")

                if scan_result.get("output_json") and action in ("search", "explain", "rename", "fix"):
                    if action in ("search", "explain"):
                        result_payload = scan_result.get("data", {})
                    else:
                        result_payload = write_result or {}
                    try:
                        payload = await asyncio.wait_for(
                            run_result_writer(llm_client, user_input, action, result_payload),
                            timeout=agent_timeout,
                        )
                    except asyncio.TimeoutError:
                        payload = {"request": user_input, "action": action, "result": result_payload}
                    out_path = get_next_result_json_path(args.project)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    print(f"\n[INFO] Result written to {out_path.name}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
    finally:
        if project_index:
            project_index.cleanup()


def main():
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(amain())


if __name__ == "__main__":
    main()
