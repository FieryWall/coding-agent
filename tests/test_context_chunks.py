"""Tests that the editor receives proper context chunks from vectorstore."""
import pytest

from coding_agent.main import _process_file_with_planner
from coding_agent.agents.change_planner import PlannedChange
from coding_agent.vectorstore.project_index import ProjectIndex
from coding_agent.vectorstore.embeddings import EmbeddingProvider
from coding_agent.tools import BackupManager


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, texts):
        return [[float(i) / self._dimension for i in range(self._dimension)] for _ in texts]

    def embed_query(self, text):
        return [float(i) / self._dimension for i in range(self._dimension)]

    @property
    def dimension(self):
        return self._dimension


JS_CODE = """\
function calculateMax(a, b) {
    return max(a, b);
}

function calculateMin(a, b) {
    return Math.min(a, b);
}
"""


class _ContextCaptureLLM:
    """LLM stub that captures what chunks the editor prompt contains."""

    def __init__(self):
        self.editor_prompts = []

    async def acomplete(self, prompt: str, agent_name: str = None, max_tokens: int = 4096) -> str:
        if agent_name == "editor":
            self.editor_prompts.append(prompt)
            return '{"changes": [{"original": "max(a, b)", "new_content": "min(a, b)"}]}'
        if agent_name == "syntax_check":
            return '{"valid": true}'
        if agent_name == "validator":
            return '{"fixable": true, "cause": "test", "suggestion": "test"}'
        return ""


@pytest.mark.asyncio
async def test_editor_receives_function_context_from_vectorstore(tmp_path):
    """When vectorstore is available, editor should get full function context,
    not just the minimal text to replace."""
    f = tmp_path / "math.js"
    f.write_text(JS_CODE, encoding="utf-8")

    index = ProjectIndex(embedding_provider=MockEmbeddingProvider())
    index.manager.add_file(str(f))

    llm = _ContextCaptureLLM()
    backup = BackupManager()
    backup.init_backup()
    backup.backup_file(str(f))

    planned = [PlannedChange(
        original="max(a, b)",
        instruction='Rename call "max" to "min"',
        skip=False,
    )]

    await _process_file_with_planner(
        llm, str(f), planned, "rename", "min", backup,
        project_index=index,
    )

    assert len(llm.editor_prompts) >= 1
    prompt = llm.editor_prompts[0]
    # Editor should see the full function context, not just "max(a, b)"
    assert "calculateMax" in prompt
    assert "function" in prompt.lower()

    backup.cleanup()


@pytest.mark.asyncio
async def test_editor_falls_back_without_vectorstore(tmp_path):
    """Without vectorstore, editor should still work with minimal chunk."""
    f = tmp_path / "math.js"
    f.write_text(JS_CODE, encoding="utf-8")

    llm = _ContextCaptureLLM()
    backup = BackupManager()
    backup.init_backup()
    backup.backup_file(str(f))

    planned = [PlannedChange(
        original="max(a, b)",
        instruction='Rename call "max" to "min"',
        skip=False,
    )]

    await _process_file_with_planner(
        llm, str(f), planned, "rename", "min", backup,
        project_index=None,
    )

    assert len(llm.editor_prompts) >= 1
    prompt = llm.editor_prompts[0]
    # Should still contain the target text
    assert "max(a, b)" in prompt

    backup.cleanup()


@pytest.mark.asyncio
async def test_editor_falls_back_when_chunk_not_found(tmp_path):
    """When vectorstore has no matching chunk, editor falls back to minimal."""
    f = tmp_path / "math.js"
    f.write_text(JS_CODE, encoding="utf-8")

    index = ProjectIndex(embedding_provider=MockEmbeddingProvider())
    # Index the file so vectorstore exists but won't match
    index.manager.add_file(str(f))

    llm = _ContextCaptureLLM()
    backup = BackupManager()
    backup.init_backup()
    backup.backup_file(str(f))

    planned = [PlannedChange(
        original="thisTextDoesNotExistAnywhere",
        instruction='Fix something',
        skip=False,
    )]

    await _process_file_with_planner(
        llm, str(f), planned, "fix", "", backup,
        project_index=index,
    )

    assert len(llm.editor_prompts) >= 1
    prompt = llm.editor_prompts[0]
    # Should contain the fallback target text
    assert "thisTextDoesNotExistAnywhere" in prompt

    backup.cleanup()
