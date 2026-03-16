import pytest
from coding_agent.vectorstore.chunker import CodeChunker


class TestChunkerJavaScript:
    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    def test_function_declaration(self, chunker):
        code = """
function greet(name) {
    return "Hello, " + name;
}
"""
        chunks = chunker.chunk_code(code, "javascript", "test.js")

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "function"
        assert chunks[0].name == "greet"
        assert "function greet" in chunks[0].content

    def test_arrow_function(self, chunker):
        code = """
const add = (a, b) => a + b;
"""
        chunks = chunker.chunk_code(code, "javascript", "test.js")

        assert len(chunks) >= 1
        variable_chunks = [c for c in chunks if c.chunk_type == "variable"]
        assert len(variable_chunks) == 1
        assert variable_chunks[0].name == "add"

    def test_class(self, chunker):
        code = """
class Calculator {
    constructor() {
        this.value = 0;
    }

    add(x) {
        this.value += x;
        return this;
    }

    getValue() {
        return this.value;
    }
}
"""
        chunks = chunker.chunk_code(code, "javascript", "test.js")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Calculator"

        method_chunks = [c for c in chunks if c.chunk_type == "function"]
        method_names = {c.name for c in method_chunks}
        assert "add" in method_names
        assert "getValue" in method_names

    def test_multiple_functions(self, chunker):
        code = """
function first() {
    return 1;
}

function second() {
    return 2;
}

function third() {
    return 3;
}
"""
        chunks = chunker.chunk_code(code, "javascript", "test.js")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 3
        names = {c.name for c in function_chunks}
        assert names == {"first", "second", "third"}
