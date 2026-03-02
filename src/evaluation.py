# evaluation.py
import numpy as np
import pandas as pd
import re
import random


# Utilities

def split_places_50_50(meta_df, seed=42):
    """
    Split place ids into 50% queries / 50% candidates.
    """
    ids = meta_df["idplace"].dropna().unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)
    mid = len(ids) // 2
    return set(ids[:mid]), set(ids[mid:])


def _to_set(val):
    """
    Convert metadata field (csv string, list, NaN) to a set of strings.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    if isinstance(val, list):
        return {str(v) for v in val}
    return {s.strip() for s in re.split(r"[|,]", str(val)) if s.strip()}


# Matching functions

def match_L1(q_pid, c_pid, meta_df):
    """
    Level 1 match: same coarse type (typeR).
    """
    if q_pid not in meta_df.index or c_pid not in meta_df.index:
        return False
    return meta_df.loc[q_pid, "typeR"] == meta_df.loc[c_pid, "typeR"]


def match_L2(q_pid, c_pid, meta_df):
    """
    Level 2 match: fine-grained metadata.
    """
    if q_pid not in meta_df.index or c_pid not in meta_df.index:
        return False

    q_row = meta_df.loc[q_pid]
    c_row = meta_df.loc[c_pid]

    t = q_row["typeR"]
    if t != c_row["typeR"]:
        return False

    # Hotel: same price range
    if t == "H":
        return q_row["priceRange"] == c_row["priceRange"]

    # Restaurant: overlapping cuisine/type
    if t == "R":
        q_tags = _to_set(q_row["restaurantTypeCuisine"])
        c_tags = _to_set(c_row["restaurantTypeCuisine"])
        return bool(q_tags & c_tags)

    # Attraction: overlapping subcategory or subtype
    if t in ["A", "AP"]:
        q_tags = _to_set(q_row["activiteSubCategorie"]) | _to_set(q_row["activiteSubType"])
        c_tags = _to_set(c_row["activiteSubCategorie"]) | _to_set(c_row["activiteSubType"])
        return bool(q_tags & c_tags)

    return False



# Ranking error

def ranking_error(query_id, ranked_ids, meta_df, level=1):
    """
    Position of the first matching candidate.
    """
    if level == 1:
        matcher = match_L1
    else:
        matcher = match_L2

    for pos, cid in enumerate(ranked_ids):
        if matcher(query_id, cid, meta_df):
            return pos
    return None


# Generic evaluation

def evaluate_ranking_model(
    query_ids,
    rank_function,
    meta_df,
    level=1,
    top_n=None,
):
    """
    Evaluate a ranking model using Ranking Error (L1 or L2).
    """
    errors = []

    for qid in query_ids:
        ranked_ids = rank_function(qid)

        if top_n is not None:
            ranked_ids = ranked_ids[:top_n]

        err = ranking_error(qid, ranked_ids, meta_df, level)
        if err is not None:
            errors.append(err)

    return {
        "mean_ranking_error": float(np.mean(errors)) if errors else float("nan"),
        "n_queries_evaluated": len(errors),
    }