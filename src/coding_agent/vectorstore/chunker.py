from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class CodeChunk:
    content: str
    file_path: str
    language: str
    chunk_type: str
    name: Optional[str]
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name or "",
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


class CodeChunker:
    LANGUAGE_MAP = {
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".py": "python",
    }

    NODE_TYPES = {
        "javascript": {
            "function": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
            "class": ["class_declaration"],
            "variable": ["lexical_declaration", "variable_declaration"],
        },
        "typescript": {
            "function": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
            "class": ["class_declaration"],
            "variable": ["lexical_declaration", "variable_declaration"],
            "interface": ["interface_declaration"],
            "type": ["type_alias_declaration"],
        },
        "python": {
            "function": ["function_definition"],
            "class": ["class_definition"],
            "variable": ["assignment"],
        },
    }

    def __init__(self):
        self._parsers = {}

    def _get_parser(self, language: str):
        if language in self._parsers:
            return self._parsers[language]

        import tree_sitter_javascript as ts_js
        import tree_sitter_python as ts_py
        from tree_sitter import Language, Parser

        languages = {  # PyCapsule arg is correct API for tree-sitter 0.25.x
            "javascript": Language(ts_js.language()),  # type: ignore[deprecated]
            "typescript": Language(ts_js.language()),  # type: ignore[deprecated]
            "python": Language(ts_py.language()),  # type: ignore[deprecated]
        }

        if language not in languages:
            raise ValueError(f"Unsupported language: {language}")

        parser = Parser(languages[language])
        self._parsers[language] = parser
        return parser

    def _detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext not in self.LANGUAGE_MAP:
            raise ValueError(f"Unsupported file extension: {ext}")
        return self.LANGUAGE_MAP[ext]

    def _get_node_name(self, node, language: str) -> Optional[str]:
        if language in ("javascript", "typescript"):
            if node.type == "function_declaration":
                for child in node.children:
                    if child.type == "identifier":
                        return child.text.decode("utf-8")
            elif node.type == "class_declaration":
                for child in node.children:
                    if child.type == "identifier":
                        return child.text.decode("utf-8")
            elif node.type == "method_definition":
                for child in node.children:
                    if child.type == "property_identifier":
                        return child.text.decode("utf-8")
            elif node.type in ("lexical_declaration", "variable_declaration"):
                for child in node.children:
                    if child.type == "variable_declarator":
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                return subchild.text.decode("utf-8")
        elif language == "python":
            if node.type == "function_definition":
                for child in node.children:
                    if child.type == "identifier":
                        return child.text.decode("utf-8")
            elif node.type == "class_definition":
                for child in node.children:
                    if child.type == "identifier":
                        return child.text.decode("utf-8")
        return None

    def _get_chunk_type(self, node_type: str, language: str) -> str:
        node_types = self.NODE_TYPES.get(language, {})
        for chunk_type, types in node_types.items():
            if node_type in types:
                return chunk_type
        return "block"

    def _extract_chunks(self, node, source: bytes, file_path: str, language: str, chunks: List[CodeChunk]):
        node_types = self.NODE_TYPES.get(language, {})
        all_types = []
        for types in node_types.values():
            all_types.extend(types)

        if node.type in all_types:
            content = source[node.start_byte:node.end_byte].decode("utf-8")
            name = self._get_node_name(node, language)
            chunk_type = self._get_chunk_type(node.type, language)

            chunks.append(CodeChunk(
                content=content,
                file_path=file_path,
                language=language,
                chunk_type=chunk_type,
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            ))

        for child in node.children:
            self._extract_chunks(child, source, file_path, language, chunks)

    def chunk_file(self, file_path: str, content: Optional[str] = None) -> List[CodeChunk]:
        if content is None:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        language = self._detect_language(file_path)
        parser = self._get_parser(language)

        source = content.encode("utf-8")
        tree = parser.parse(source)

        chunks = []
        self._extract_chunks(tree.root_node, source, file_path, language, chunks)

        return chunks

    def chunk_code(self, code: str, language: str, file_path: str = "<string>") -> List[CodeChunk]:
        parser = self._get_parser(language)
        source = code.encode("utf-8")
        tree = parser.parse(source)

        chunks = []
        self._extract_chunks(tree.root_node, source, file_path, language, chunks)

        return chunks
