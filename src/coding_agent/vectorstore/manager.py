from typing import List, Optional
from dataclasses import dataclass
import tempfile
import shutil
import uuid

import lancedb
import pyarrow as pa

from .embeddings import EmbeddingProvider
from .chunker import CodeChunk, CodeChunker


@dataclass
class SearchResult:
    content: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    score: float


class VectorStoreManager:
    TABLE_NAME = "code_chunks"

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.chunker = CodeChunker()
        self._db_path = tempfile.mkdtemp(prefix=f"lancedb_{uuid.uuid4().hex[:8]}_")
        self.db = lancedb.connect(self._db_path)
        self._table = None

    def _get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.embedding_provider.dimension)),
            pa.field("content", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("language", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("name", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
        ])

    def _ensure_table(self):
        if self._table is None:
            if self.TABLE_NAME in self.db.list_tables().tables:
                self._table = self.db.open_table(self.TABLE_NAME)
            else:
                self._table = self.db.create_table(
                    self.TABLE_NAME,
                    schema=self._get_schema(),
                )

    def reset(self):
        if self.TABLE_NAME in self.db.list_tables().tables:
            self.db.drop_table(self.TABLE_NAME)
        self._table = None

    def cleanup(self):
        shutil.rmtree(self._db_path, ignore_errors=True)

    def add_file(self, file_path: str, content: Optional[str] = None) -> int:
        chunks = self.chunker.chunk_file(file_path, content)
        return self._add_chunks(chunks)

    def add_code(self, code: str, language: str, file_path: str = "<string>") -> int:
        chunks = self.chunker.chunk_code(code, language, file_path)
        return self._add_chunks(chunks)

    def _add_chunks(self, chunks: List[CodeChunk]) -> int:
        if not chunks:
            return 0

        self._ensure_table()

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed(texts)

        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "vector": embedding,
                "content": chunk.content,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name or "",
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            })

        self._table.add(records)
        return len(records)

    def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> List[SearchResult]:
        self._ensure_table()

        if self._table.count_rows() == 0:
            return []

        query_embedding = self.embedding_provider.embed_query(query)

        search_query = self._table.search(query_embedding)

        filters = []
        if language:
            filters.append(f"language = '{language}'")
        if chunk_type:
            filters.append(f"chunk_type = '{chunk_type}'")
        if file_path:
            filters.append(f"file_path = '{file_path}'")

        if filters:
            search_query = search_query.where(" AND ".join(filters))

        results = search_query.limit(limit).to_list()

        return [
            SearchResult(
                content=r["content"],
                file_path=r["file_path"],
                language=r["language"],
                chunk_type=r["chunk_type"],
                name=r["name"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=1 - r["_distance"],
            )
            for r in results
        ]

    def search_similar_code(self, code_snippet: str, limit: int = 5) -> List[SearchResult]:
        return self.search(code_snippet, limit=limit)

    def get_chunks_by_file(self, file_path: str) -> List[SearchResult]:
        self._ensure_table()

        if self._table.count_rows() == 0:
            return []

        results = self._table.search().where(f"file_path = '{file_path}'").limit(1000).to_list()

        return [
            SearchResult(
                content=r["content"],
                file_path=r["file_path"],
                language=r["language"],
                chunk_type=r["chunk_type"],
                name=r["name"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                score=1.0,
            )
            for r in results
        ]

    def count_chunks(self) -> int:
        self._ensure_table()
        return self._table.count_rows()
