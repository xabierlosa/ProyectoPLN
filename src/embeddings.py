from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedder(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def _prefix_texts(texts: List[str], prefix: str) -> List[str]:
    return [f"{prefix}{text}" for text in texts]


def embed_passages(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    prefixed = _prefix_texts(texts, "passage: ")
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    prefixed = f"query: {query}"
    vectors = model.encode([prefixed], normalize_embeddings=True)
    return vectors[0]
