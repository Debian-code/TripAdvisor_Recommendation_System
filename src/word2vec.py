from gensim.models import Word2Vec
import processing
import numpy as np
import pandas as pd
from typing import List, Tuple
import score



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

    corpus_tokens = processing.get_corpus_tokens(reviews)
    review_vecs = np.vstack([get_review_vector(toks, model) for toks in corpus_tokens]).astype(np.float32)

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

def evaluate(
    reviews: pd.DataFrame,
    model: Word2Vec,
    pooling: str = "mean",
    level: int = 1,
    random_state: int = 42,
) -> dict:
    """
    Split aléatoire 50/50 des places (sur idplace unique).
    
    Level 1 : match sur typeR
    Level 2 : match sur métadonnées fines selon typeR
        - Attraction (A)  : activiteSubCategorie ou activiteSubType (≥1 en commun)
        - Restaurant (R)  : restaurantType ou cuisine (≥1 en commun)
        - Hotel (H)       : priceRange (égalité)
        - AP              : fallback level-1

    Métriques retournées :
        mean_cos_sim          : cosine similarity moyenne du top-1 par query
        max_cos_sim           : cosine similarity max du top-1 par query
        eval_score            : % de queries dont le top-1 matche (binaire)
        ranking_error         : position 0-based du 1er match, moyennée sur les
                                queries ayant au moins 1 match dans le test set
        n_queries             : nombre de queries (train)
        n_queries_with_match  : queries avec au moins 1 match possible dans test
    """
    import random

    # ── 1. construire les métadonnées par place (1 ligne par idplace) ───────
    # On agrège les colonnes meta au niveau place (on prend la 1ère valeur)
    meta_cols = ["idplace", "typeR", "activiteSubCategorie", "activiteSubType",
                 "priceRange", "restaurantType"]
    available_cols = [c for c in meta_cols if c in reviews.columns]
    
    place_meta = reviews[available_cols].drop_duplicates(subset="idplace").set_index("idplace")

    # ── 2. split 50/50 sur les idplace ──────────────────────────────────────
    all_ids = place_meta.index.tolist()
    random.seed(random_state)
    random.shuffle(all_ids)
    mid        = len(all_ids) // 2
    train_ids  = all_ids[:mid]
    test_ids   = all_ids[mid:]

    # ── 3. vecteurs par place ────────────────────────────────────────────────
    place_ids_arr, place_vecs_arr = get_place_vectors(reviews, model, pooling=pooling)
    id2vec = {pid: place_vecs_arr[i] for i, pid in enumerate(place_ids_arr)}

    train_list = [pid for pid in train_ids if pid in id2vec]
    test_list  = [pid for pid in test_ids  if pid in id2vec]

    if not train_list or not test_list:
        raise ValueError("Split trop petit : pas assez de places avec vecteurs.")

    test_vecs = np.vstack([id2vec[pid] for pid in test_list]).astype(np.float32)

    # ── helper : extraire les tags level-2 d'une place ──────────────────────
    def to_set(val) -> set:
        """Convertit une valeur (str, list, None) en set de strings."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return set()
        if isinstance(val, list):
            return {str(v) for v in val}
        return {str(val)}

    def get_l2_tags(pid) -> set:
        if pid not in place_meta.index:
            return set()
        row = place_meta.loc[pid]
        tR  = row.get("typeR", "")
        if tR == "A":
            return to_set(row.get("activiteSubCategorie")) | to_set(row.get("activiteSubType"))
        elif tR == "R":
            return to_set(row.get("restaurantType"))
        elif tR == "H":
            return to_set(row.get("priceRange"))
        return set()  # AP ou inconnu → pas de tags fins

    # ── helper : match entre query et candidat ───────────────────────────────
    def matches(q_pid, c_pid) -> bool:
        if q_pid not in place_meta.index or c_pid not in place_meta.index:
            return False
        q_type = place_meta.loc[q_pid, "typeR"]
        c_type = place_meta.loc[c_pid, "typeR"]

        if level == 1:
            return q_type == c_type

        # level 2
        if q_type != c_type:
            return False  # types différents → pas de match
        q_tags = get_l2_tags(q_pid)
        c_tags = get_l2_tags(c_pid)
        if not q_tags or not c_tags:
            return True  # pas de métadonnées fines → on accepte le match level-1
        return bool(q_tags & c_tags)  # au moins 1 tag en commun

    # ── 4. boucle d'évaluation ───────────────────────────────────────────────
    top1_sims      = []
    binary_hits    = []
    ranking_errors = []

    for q_pid in train_list:
        q_vec = id2vec[q_pid]

        sims       = np.array([score.cos_sim(q_vec, v) for v in test_vecs], dtype=np.float32)
        ranked_idx = np.argsort(sims)[::-1]

        top1_sims.append(float(sims[ranked_idx[0]]))

        # cherche la 1ère position avec un match
        first_match_pos = None
        for pos, idx in enumerate(ranked_idx):
            if matches(q_pid, test_list[idx]):
                first_match_pos = pos
                break

        if first_match_pos is None:
            # aucun match possible dans le test set → query ignorée pour ranking_error
            pass
        else:
            binary_hits.append(1 if first_match_pos == 0 else 0)
            ranking_errors.append(first_match_pos)  # 0-based, erreur = n-1

    return {
        "mean_cos_sim":         float(np.mean(top1_sims))    if top1_sims      else float("nan"),
        "max_cos_sim":          float(np.max(top1_sims))     if top1_sims      else float("nan"),
        "eval_score":           float(np.mean(binary_hits))  if binary_hits    else float("nan"),
        "ranking_error":        float(np.mean(ranking_errors)) if ranking_errors else float("nan"),
        "n_queries":            len(train_list),
        "n_queries_with_match": len(ranking_errors),
    }