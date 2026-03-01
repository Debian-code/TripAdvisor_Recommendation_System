from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_corpus_by_place(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    grouped = df.groupby("idplace")["review"].apply(lambda x: " ".join(map(str, x)))
    place_texts = grouped.tolist()
    place_ids = grouped.index.to_numpy()
    return place_texts, place_ids


def encode_places(
    place_texts: List[str],
    model: SentenceTransformer = model,
) -> np.ndarray:
    place_embeddings = model.encode(
        place_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return place_embeddings.astype(np.float32)


def score_sbert(
    query: str,
    place_embeddings: np.ndarray,
    model: SentenceTransformer = model,
) -> np.ndarray:
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # place_embeddings: (n_places, dim), query_embedding: (dim,)
    sims = place_embeddings @ query_embedding  # (n_places,)
    return sims


def top_k_sbert(
    query: str,
    place_ids: np.ndarray,
    place_embeddings: np.ndarray,
    model: SentenceTransformer = model,
    k: int = 10,
) -> pd.DataFrame:
    sims = score_sbert(query, place_embeddings, model=model)
    k = min(k, sims.shape[0])

    top_idx = np.argsort(sims)[::-1][:k]
    return pd.DataFrame(
        {
            "idplace": place_ids[top_idx],
            "score": sims[top_idx],
        }
    )