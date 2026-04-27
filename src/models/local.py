import warnings

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel

warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")


class LocalEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self._model.encode(texts, show_progress_bar=False))
