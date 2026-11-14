"""Real embedding adapter using Google Gemini API (no mock fallback)."""
import time
from typing import Protocol

from google import genai
from google.genai import types


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Google Gemini embedding adapter using RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: str | None = None,
        batch_size: int = 32,
        dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required; mock fallback is prohibited.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.client = genai.Client(api_key=api_key)
        self.api_calls = 0
        self.embedded_chars = 0

        if dimensions is None:
            probe = self.embed_query("dimension probe")
            # Reset counters after initialization probe
            self.api_calls = 0
            self.embedded_chars = 0
            self.dimensions = len(probe)
        else:
            self.dimensions = dimensions

    def _call_with_retry(self, contents: list[str] | str, task_type: str) -> list[list[float]]:
        # ponytail: fixed 4-attempt backoff for transient rate limits; upgrade to adaptive limiter if corpus scales
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                self.api_calls += 1
                if isinstance(contents, str):
                    self.embedded_chars += len(contents)
                else:
                    self.embedded_chars += sum(len(t) for t in contents)
                return [list(emb.values) for emb in resp.embeddings]
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"Gemini embedding request failed after retries: {last_err}") from last_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vecs = self._call_with_retry(batch, task_type="RETRIEVAL_DOCUMENT")
            if len(batch_vecs) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} vectors, got {len(batch_vecs)}")
            vectors.extend(batch_vecs)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        vecs = self._call_with_retry(question, task_type="RETRIEVAL_QUERY")
        if len(vecs) != 1:
            raise RuntimeError(f"Expected 1 query vector, got {len(vecs)}")
        return vecs[0]
