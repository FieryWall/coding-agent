import pytest
from coding_agent.vectorstore.chunker import CodeChunker


class TestChunkerTypeScript:
    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    def test_typed_function(self, chunker):
        code = """
function greet(name: string): string {
    return "Hello, " + name;
}
"""
        chunks = chunker.chunk_code(code, "typescript", "test.ts")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].name == "greet"

    def test_class_with_types(self, chunker):
        code = """
class User {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    getName(): string {
        return this.name;
    }
}
"""
        chunks = chunker.chunk_code(code, "typescript", "test.ts")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "User"
