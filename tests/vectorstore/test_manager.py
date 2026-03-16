import pytest
from unittest.mock import MagicMock
from coding_agent.vectorstore.manager import VectorStoreManager, SearchResult
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


class TestVectorStoreManager:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(MockEmbeddingProvider())

    def test_add_javascript_code(self, manager):
        code = """
function greet(name) {
    return "Hello, " + name;
}

function farewell(name) {
    return "Goodbye, " + name;
}
"""
        count = manager.add_code(code, "javascript", "test.js")
        assert count == 2
        assert manager.count_chunks() == 2

    def test_add_python_code(self, manager):
        code = """
def greet(name):
    return f"Hello, {name}"

class Greeter:
    def __init__(self, prefix):
        self.prefix = prefix

    def greet(self, name):
        return f"{self.prefix} {name}"
"""
        count = manager.add_code(code, "python", "test.py")
        assert count >= 3
        assert manager.count_chunks() >= 3

    def test_add_typescript_code(self, manager):
        code = """
function add(a: number, b: number): number {
    return a + b;
}

class Calculator {
    value: number = 0;

    add(x: number): Calculator {
        this.value += x;
        return this;
    }
}
"""
        count = manager.add_code(code, "typescript", "test.ts")
        assert count >= 2
        assert manager.count_chunks() >= 2

    def test_reset(self, manager):
        manager.add_code("function test() {}", "javascript", "test.js")
        assert manager.count_chunks() == 1

        manager.reset()
        assert manager.count_chunks() == 0

    def test_search_returns_results(self, manager):
        manager.add_code("""
function greet(name) {
    return "Hello, " + name;
}
""", "javascript", "test.js")

        results = manager.search("greeting function")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert "greet" in results[0].content

    def test_search_empty_store(self, manager):
        results = manager.search("anything")
        assert results == []

    def test_search_with_language_filter(self, manager):
        manager.add_code("function jsFunc() {}", "javascript", "test.js")
        manager.add_code("def pyFunc(): pass", "python", "test.py")

        results = manager.search("function", language="javascript")
        for r in results:
            assert r.language == "javascript"

    def test_search_with_chunk_type_filter(self, manager):
        manager.add_code("""
class MyClass {
    method() {}
}
function standalone() {}
""", "javascript", "test.js")

        results = manager.search("code", chunk_type="class")
        for r in results:
            assert r.chunk_type == "class"

    def test_search_with_file_path_filter(self, manager):
        manager.add_code("function a() {}", "javascript", "file1.js")
        manager.add_code("function b() {}", "javascript", "file2.js")

        results = manager.search("function", file_path="file1.js")
        for r in results:
            assert r.file_path == "file1.js"

    def test_get_chunks_by_file(self, manager):
        manager.add_code("function a() {}", "javascript", "file1.js")
        manager.add_code("function b() {}", "javascript", "file2.js")

        chunks = manager.get_chunks_by_file("file1.js")
        assert len(chunks) == 1
        assert chunks[0].file_path == "file1.js"

    def test_search_similar_code(self, manager):
        manager.add_code("""
function calculateSum(a, b) {
    return a + b;
}

function calculateProduct(a, b) {
    return a * b;
}
""", "javascript", "math.js")

        results = manager.search_similar_code("function add(x, y) { return x + y; }")
        assert len(results) >= 1

    def test_search_result_has_score(self, manager):
        manager.add_code("function test() {}", "javascript", "test.js")

        results = manager.search("test")
        assert len(results) >= 1
        assert hasattr(results[0], "score")
        assert isinstance(results[0].score, float)


class TestVectorStoreManagerMultiLanguage:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(MockEmbeddingProvider())

    def test_mixed_language_indexing(self, manager):
        manager.add_code("function jsFunc() {}", "javascript", "app.js")
        manager.add_code("def py_func(): pass", "python", "app.py")
        manager.add_code("function tsFunc(): void {}", "typescript", "app.ts")

        assert manager.count_chunks() == 3

    def test_search_across_languages(self, manager):
        manager.add_code("function greet(name) { return 'Hello ' + name; }", "javascript", "app.js")
        manager.add_code("def greet(name): return f'Hello {name}'", "python", "app.py")

        results = manager.search("greeting function", limit=10)
        assert len(results) >= 2

        languages = {r.language for r in results}
        assert "javascript" in languages or "python" in languages


class TestVectorStoreManagerEdgeCases:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(MockEmbeddingProvider())

    def test_empty_code(self, manager):
        count = manager.add_code("", "javascript", "empty.js")
        assert count == 0

    def test_code_without_functions(self, manager):
        code = "const x = 1; const y = 2;"
        count = manager.add_code(code, "javascript", "vars.js")
        assert count >= 0

    def test_large_file_chunking(self, manager):
        functions = [f"function func{i}() {{ return {i}; }}" for i in range(50)]
        code = "\n".join(functions)

        count = manager.add_code(code, "javascript", "large.js")
        assert count == 50
        assert manager.count_chunks() == 50
