import pytest
from coding_agent.vectorstore.chunker import CodeChunker


class TestChunkerMetadata:
    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    def test_line_numbers(self, chunker):
        code = """line1
line2
def test():
    pass
line5
"""
        chunks = chunker.chunk_code(code, "python", "test.py")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].start_line == 3
        assert function_chunks[0].end_line == 4

    def test_file_path_preserved(self, chunker):
        code = "def test(): pass"
        chunks = chunker.chunk_code(code, "python", "/path/to/file.py")

        assert len(chunks) == 1
        assert chunks[0].file_path == "/path/to/file.py"

    def test_language_preserved(self, chunker):
        code = "function test() {}"
        chunks = chunker.chunk_code(code, "javascript", "test.js")

        assert len(chunks) == 1
        assert chunks[0].language == "javascript"
