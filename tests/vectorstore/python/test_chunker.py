import pytest
from coding_agent.vectorstore.chunker import CodeChunker


class TestChunkerPython:
    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    def test_function(self, chunker):
        code = """
def greet(name):
    return f"Hello, {name}"
"""
        chunks = chunker.chunk_code(code, "python", "test.py")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].name == "greet"

    def test_class(self, chunker):
        code = """
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self

    def get_value(self):
        return self.value
"""
        chunks = chunker.chunk_code(code, "python", "test.py")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Calculator"

        method_chunks = [c for c in chunks if c.chunk_type == "function"]
        method_names = {c.name for c in method_chunks}
        assert "__init__" in method_names
        assert "add" in method_names
        assert "get_value" in method_names

    def test_async_function(self, chunker):
        code = """
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
"""
        chunks = chunker.chunk_code(code, "python", "test.py")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].name == "fetch_data"

    def test_decorated_function(self, chunker):
        code = """
@decorator
def decorated_function():
    pass
"""
        chunks = chunker.chunk_code(code, "python", "test.py")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].name == "decorated_function"
