from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from .manager import VectorStoreManager, SearchResult
from .embeddings import EmbeddingProvider, OpenAIEmbedding
from .chunker import CodeChunker


SUPPORTED_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".py"}


@dataclass
class RelevantChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: Optional[str]


class ProjectIndex:
    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or OpenAIEmbedding()
        self.manager = VectorStoreManager(self.embedding_provider)
        self.chunker = CodeChunker()
        self._indexed_files: set[str] = set()

    def index_project(self, project_path: str) -> int:
        self.manager.reset()
        self._indexed_files.clear()
        
        total_chunks = 0
        project = Path(project_path)
        
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in project.rglob(f"*{ext}"):
                if "node_modules" in file_path.parts:
                    continue
                if "__pycache__" in file_path.parts:
                    continue
                
                try:
                    count = self.manager.add_file(str(file_path))
                    total_chunks += count
                    self._indexed_files.add(str(file_path))
                except Exception:
                    pass
        
        return total_chunks

    def index_file(self, file_path: str) -> int:
        if file_path in self._indexed_files:
            return 0
        
        try:
            count = self.manager.add_file(file_path)
            self._indexed_files.add(file_path)
            return count
        except Exception:
            return 0

    def get_relevant_chunks(
        self,
        task_description: str,
        target_files: List[str],
        limit_per_file: int = 10,
    ) -> dict[str, List[RelevantChunk]]:
        result: dict[str, List[RelevantChunk]] = {}
        
        for file_path in target_files:
            file_chunks = self.manager.get_chunks_by_file(file_path)
            
            if not file_chunks:
                continue
            
            semantic_results = self.manager.search(
                task_description,
                limit=limit_per_file,
                file_path=file_path,
            )
            
            relevant_names = set()
            for r in semantic_results:
                if r.name:
                    relevant_names.add(r.name)
            
            all_relevant: dict[tuple, SearchResult] = {}
            
            for r in semantic_results:
                key = (r.file_path, r.start_line, r.end_line)
                all_relevant[key] = r
            
            for chunk in file_chunks:
                if self._references_any(chunk.content, relevant_names):
                    key = (chunk.file_path, chunk.start_line, chunk.end_line)
                    if key not in all_relevant:
                        all_relevant[key] = chunk
            
            chunks = [
                RelevantChunk(
                    content=r.content,
                    file_path=r.file_path,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    chunk_type=r.chunk_type,
                    name=r.name if r.name else None,
                )
                for r in all_relevant.values()
            ]
            
            chunks.sort(key=lambda c: c.start_line)
            result[file_path] = chunks
        
        return result

    def _references_any(self, content: str, names: set[str]) -> bool:
        for name in names:
            if name and name in content:
                return True
        return False

    def get_chunks_containing(
        self,
        file_path: str,
        text: str,
        max_chunks: int = 3,
    ) -> List[RelevantChunk]:
        """Find chunks from the vectorstore that contain the given text.

        Used to provide semantic context (full function/class) to the editor
        instead of just the minimal text to replace.
        """
        file_chunks = self.manager.get_chunks_by_file(file_path)
        if not file_chunks:
            return []

        matching = []
        for chunk in file_chunks:
            if text in chunk.content:
                matching.append(RelevantChunk(
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    name=chunk.name if chunk.name else None,
                ))

        matching.sort(key=lambda c: len(c.content))
        return matching[:max_chunks]

    def search_by_name(self, name: str, limit: int = 10) -> List[SearchResult]:
        return self.manager.search(name, limit=limit)

    def cleanup(self):
        self.manager.cleanup()
