"""Stand-in embedders so tests never download or run the real model."""

import numpy as np

from ml.embeddings import EmbeddingUnavailableError


class FakeEmbedder:
    """Marks which known words appear in each text (a tiny fake 'meaning')."""

    VOCAB = ["python", "sql", "docker", "chef", "pastry"]

    def encode(self, texts):
        return np.array(
            [[1.0 if word in text.lower() else 0.0 for word in self.VOCAB]
             for text in texts]
        )


class FailingEmbedder:
    def encode(self, texts):
        raise EmbeddingUnavailableError("simulated failure")
