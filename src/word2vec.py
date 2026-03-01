from gensim.models import Word2Vec
import processing
import numpy as np
import pandas as pd
from typing import List, Tuple
import score


# -----------------------------
# Train Word2Vec
# -----------------------------
def train_word2vec_model(reviews: pd.DataFrame) -> Word2Vec:
    corpus = processing.get_corpus_tokens(reviews)  # list[list[str]]
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=200,
        window=5,
        min_count=5,
        workers=4,
        sg=1  # skip-gram
    )
    return w2v


# -----------------------------
# Review vectors
# -----------------------------
def get_review_vector(tokens: List[str], model: Word2Vec, pooling: str = "mean") -> np.ndarray:
    vecs = [model.wv[t] for t in tokens if t in model.wv]

    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)

    mat = np.vstack(vecs).astype(np.float32)  # (n_tokens_in_vocab, dim)

    if pooling == "mean":
        return mat.mean(axis=0)
    elif pooling == "sum":
        return mat.sum(axis=0)
    else:
        raise ValueError("pooling must be 'mean' or 'sum'")


def get_corpus_vectors(reviews: pd.DataFrame, model: Word2Vec) -> np.ndarray:
    corpus_tokens = processing.get_corpus_tokens(reviews)
    vectors = np.vstack([get_review_vector(tokens, model) for tokens in corpus_tokens]).astype(np.float32)
    return vectors  # (n_reviews, dim)


def w2v_query_vector(query: str, model: Word2Vec) -> np.ndarray:
    q_tokens = processing.tokenize(query)
    return get_review_vector(q_tokens, model)


def w2v_scores_to_corpus(query_vec: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
    # suppose que score.cos_sim(a,b) calcule cosinus entre 2 vecteurs
    # Ici on vectorise pour éviter une boucle python si tu veux:
    # mais on reste simple et stable.
    sims = np.array([score.cos_sim(query_vec, v) for v in corpus_vectors], dtype=np.float32)
    return sims


def w2v_top_k_reviews(
    reviews: pd.DataFrame,
    query: str,
    corpus_vectors: np.ndarray,
    model: Word2Vec,
    k: int = 10
) -> pd.DataFrame:
    q_vec = w2v_query_vector(query, model)
    sims = w2v_scores_to_corpus(q_vec, corpus_vectors)
    top_idx = np.argsort(sims)[::-1][:k]

    out = reviews.iloc[top_idx].copy()
    out["score"] = sims[top_idx]
    return out[["score", "review"]]


# -----------------------------
# Place vectors (groupby idplace)
# -----------------------------
def pool_vectors(vectors: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """
    vectors: (n_reviews_place, dim)
    pooling: "mean" | "sum" | "max"
    """
    if vectors.size == 0:
        raise ValueError("No vectors to pool")

    if pooling == "mean":
        return vectors.mean(axis=0)
    if pooling == "sum":
        return vectors.sum(axis=0)
    if pooling == "max":
        return vectors.max(axis=0)

    raise ValueError("pooling must be 'mean', 'sum', or 'max'")


def get_place_vectors(
    reviews: pd.DataFrame,
    model: Word2Vec,
    place_col: str = "idplace",
    pooling: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      place_ids: (n_places,)
      place_vectors: (n_places, dim)
    """
    # 1) review vectors (même ordre que reviews)
    corpus_tokens = processing.get_corpus_tokens(reviews)
    review_vecs = np.vstack([get_review_vector(toks, model) for toks in corpus_tokens]).astype(np.float32)

    # 2) pool par place
    place_ids = []
    place_vecs = []

    for pid, idx in reviews.groupby(place_col).indices.items():
        vecs = review_vecs[np.array(idx)]
        place_ids.append(pid)
        place_vecs.append(pool_vectors(vecs, pooling=pooling))

    return np.array(place_ids), np.vstack(place_vecs).astype(np.float32)


def w2v_top_k_places(
    query: str,
    place_ids: np.ndarray,
    place_vectors: np.ndarray,
    model: Word2Vec,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    q_vec = w2v_query_vector(query, model)
    sims = np.array([score.cos_sim(q_vec, v) for v in place_vectors], dtype=np.float32)
    top_idx = np.argsort(sims)[::-1][:k]
    return place_ids[top_idx], sims[top_idx]