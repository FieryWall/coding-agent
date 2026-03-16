from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    p = PROMPTS_DIR / f"{name}.txt" if not name.endswith(".txt") else PROMPTS_DIR / name
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def load_scanner_prompt(step: str, user_input: str, project_path: str, context: str = "") -> str:
    main = (PROMPTS_DIR / "scanner" / "main.txt").read_text(encoding="utf-8")
    step_instructions = (PROMPTS_DIR / "scanner" / f"step_{step}.txt").read_text(encoding="utf-8")
    return main.replace("{step_instructions}", step_instructions).replace(
        "{user_input}", user_input
    ).replace("{project_path}", project_path).replace("{context}", context)
