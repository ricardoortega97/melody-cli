from abc import ABC, abstractmethod
import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a 2-D float32 array of shape (len(texts), embedding_dim)."""
        ...
