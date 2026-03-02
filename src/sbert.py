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


def evaluate_sbert(
    reviews: pd.DataFrame,
    sbert_model: SentenceTransformer = model,
    level: int = 1,
    random_state: int = 42,
) -> dict:
    """
    Évaluation SBERT avec split 50/50 sur les idplace.

    Level 1 : match sur typeR
    Level 2 : match sur métadonnées fines
        - A  : activiteSubCategorie ou activiteSubType (≥1 en commun)
        - R  : restaurantTypeCuisine (≥1 id en commun)
        - H  : priceRange (égalité)
        - AP : fallback level-1
    """
    import random

    # ── 1. métadonnées par place ─────────────────────────────────────────────
    meta_cols = ["idplace", "typeR", "activiteSubCategorie", "activiteSubType",
                 "restaurantTypeCuisine", "priceRange"]
    available_cols = [c for c in meta_cols if c in reviews.columns]
    place_meta = (
        reviews[available_cols]
        .drop_duplicates(subset="idplace")
        .set_index("idplace")
    )

    # ── 2. split 50/50 ───────────────────────────────────────────────────────
    all_ids = place_meta.index.tolist()
    random.seed(random_state)
    random.shuffle(all_ids)
    mid       = len(all_ids) // 2
    train_ids = set(all_ids[:mid])
    test_ids  = set(all_ids[mid:])

    # ── 3. encodage SBERT par place ──────────────────────────────────────────
    place_texts, place_ids_arr = get_corpus_by_place(reviews)
    place_embeddings = encode_places(place_texts, sbert_model)
    id2vec = {pid: place_embeddings[i] for i, pid in enumerate(place_ids_arr)}

    train_list = [pid for pid in train_ids if pid in id2vec]
    test_list  = [pid for pid in test_ids  if pid in id2vec]

    if not train_list or not test_list:
        raise ValueError("Split trop petit : pas assez de places avec embeddings.")

    test_vecs = np.vstack([id2vec[pid] for pid in test_list]).astype(np.float32)

    # ── helpers ──────────────────────────────────────────────────────────────
    def to_set(val) -> set:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return set()
        if isinstance(val, list):
            return {str(v) for v in val}
        # cas "4617,10649,10654" → split sur la virgule
        return {s.strip() for s in str(val).split(",")}

    def get_l2_tags(pid) -> set:
        if pid not in place_meta.index:
            return set()
        row = place_meta.loc[pid]
        tR  = row.get("typeR", "")
        if tR == "A":
            return to_set(row.get("activiteSubCategorie")) | to_set(row.get("activiteSubType"))
        elif tR == "R":
            return to_set(row.get("restaurantTypeCuisine"))
        elif tR == "H":
            return to_set(row.get("priceRange"))
        return set()

    def matches(q_pid, c_pid) -> bool:
        if q_pid not in place_meta.index or c_pid not in place_meta.index:
            return False
        q_type = place_meta.loc[q_pid, "typeR"]
        c_type = place_meta.loc[c_pid, "typeR"]
        if level == 1:
            return q_type == c_type
        if q_type != c_type:
            return False
        q_tags = get_l2_tags(q_pid)
        c_tags = get_l2_tags(c_pid)
        if not q_tags or not c_tags:
            return True   # pas de métadonnées fines → fallback level-1
        return bool(q_tags & c_tags)

    # ── 4. évaluation ────────────────────────────────────────────────────────
    top1_sims      = []
    binary_hits    = []
    ranking_errors = []

    for q_pid in train_list:
        q_vec = id2vec[q_pid]

        # embeddings normalisés → cosine sim = produit scalaire
        sims       = test_vecs @ q_vec          # (n_test,)
        ranked_idx = np.argsort(sims)[::-1]

        top1_sims.append(float(sims[ranked_idx[0]]))

        first_match_pos = None
        for pos, idx in enumerate(ranked_idx):
            if matches(q_pid, test_list[idx]):
                first_match_pos = pos
                break

        if first_match_pos is not None:
            binary_hits.append(1 if first_match_pos == 0 else 0)
            ranking_errors.append(first_match_pos)

    return {
        "mean_cos_sim":          float(np.mean(top1_sims))      if top1_sims      else float("nan"),
        "max_cos_sim":           float(np.max(top1_sims))       if top1_sims      else float("nan"),
        "eval_score":            float(np.mean(binary_hits))    if binary_hits    else float("nan"),
        "ranking_error":         float(np.mean(ranking_errors)) if ranking_errors else float("nan"),
        "n_queries":             len(train_list),
        "n_queries_with_match":  len(ranking_errors),
    }