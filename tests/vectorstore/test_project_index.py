import pytest
from coding_agent.vectorstore.project_index import ProjectIndex, RelevantChunk
from coding_agent.vectorstore.embeddings import EmbeddingProvider


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

const MAX_DEPTH = 10;
"""


class TestGetChunksContaining:
    @pytest.fixture
    def index(self):
        idx = ProjectIndex(embedding_provider=MockEmbeddingProvider())
        return idx

    def test_finds_chunk_containing_text(self, index, tmp_path):
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        index.manager.add_file(str(f))

        chunks = index.get_chunks_containing(str(f), "max(a, b)")
        assert len(chunks) >= 1
        assert any("calculateMax" in c.content for c in chunks)

    def test_returns_empty_for_missing_text(self, index, tmp_path):
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        index.manager.add_file(str(f))

        chunks = index.get_chunks_containing(str(f), "nonExistentFunction()")
        assert chunks == []

    def test_returns_empty_for_unknown_file(self, index):
        chunks = index.get_chunks_containing("/no/such/file.js", "max")
        assert chunks == []

    def test_returns_smallest_chunk_first(self, index, tmp_path):
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        index.manager.add_file(str(f))

        chunks = index.get_chunks_containing(str(f), "max(a, b)")
        if len(chunks) > 1:
            assert len(chunks[0].content) <= len(chunks[1].content)

    def test_max_chunks_limit(self, index, tmp_path):
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        index.manager.add_file(str(f))

        chunks = index.get_chunks_containing(str(f), "max(a, b)", max_chunks=1)
        assert len(chunks) <= 1

    def test_chunk_has_correct_fields(self, index, tmp_path):
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        index.manager.add_file(str(f))

        chunks = index.get_chunks_containing(str(f), "calculateMax")
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert isinstance(chunk, RelevantChunk)
        assert chunk.file_path == str(f)
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.chunk_type in ("function", "class", "variable", "block")

    def test_fallback_when_no_chunks_indexed(self, index, tmp_path):
        """When file exists but was never indexed, returns empty."""
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        # Don't index the file
        chunks = index.get_chunks_containing(str(f), "calculateMax")
        assert chunks == []


class TestGetRelevantChunks:
    @pytest.fixture
    def index(self):
        return ProjectIndex(embedding_provider=MockEmbeddingProvider())

    def test_returns_chunks_for_indexed_file(self, index, tmp_path):
        f = tmp_path / "math.js"
        f.write_text(JS_CODE, encoding="utf-8")
        index.manager.add_file(str(f))

        result = index.get_relevant_chunks("rename max", [str(f)])
        assert str(f) in result
        assert len(result[str(f)]) >= 1

    def test_returns_empty_for_unindexed_file(self, index, tmp_path):
        result = index.get_relevant_chunks("rename max", ["/no/file.js"])
        assert result == {}
