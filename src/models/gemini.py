import os

import numpy as np
from google import genai

from .base import EmbeddingModel

GEMINI_MODEL = "text-embedding-004"


class GeminiEmbeddingModel(EmbeddingModel):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable Gemini embeddings."
            )
        self._client = genai.Client(api_key=api_key)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            r = self._client.models.embed_content(model=GEMINI_MODEL, contents=text)
            vectors.append(r.embeddings[0].values)
        return np.array(vectors, dtype=np.float32)
