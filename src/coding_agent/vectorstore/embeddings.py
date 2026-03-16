from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._dimension = 1536

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
