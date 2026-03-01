from rank_bm25 import BM25Okapi 
import processing
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

def bm25_model(reviews : pd.DataFrame) -> BM25Okapi:
    corpus = processing.get_corpus_tokens(reviews)
    bm25 = BM25Okapi(corpus)
    return bm25

def bm25_score(query: str, bm25_model: BM25Okapi) -> float:
    scores=bm25_model.get_scores(processing.tokenize(query))
    return scores

def bm25_top_k(query : str, bm25_model : BM25Okapi,  k=10) -> pd.DataFrame:
    top_k_reviews=bm25_score(query, bm25_model).argsort()[::-1][:k]
    return top_k_reviews

def get_the_top_k_reviews(reviews : pd.DataFrame, query : str, bm25_model : BM25Okapi, k=10) -> pd.DataFrame:
    top_k_idx=bm25_top_k(query, bm25_model, k)
    top_k_reviews=reviews.iloc[top_k_idx][["review"]]
    return top_k_reviews

def bm25_fit(place_docs_df):
    """
    place_docs_df: columns [idplace, place_document]
    returns: bm25_model, doc_ids
    """
    doc_ids = place_docs_df["idplace"].tolist()
    tokenized_docs = [str(doc).split() for doc in place_docs_df["place_document"].fillna("").tolist()]
    model = BM25Okapi(tokenized_docs)
    return model, doc_ids

def bm25_rank(model, doc_ids, query_text, top_n=100, exclude_id=None):
    """
    Returns list of (idplace, score) sorted desc.
    """
    q_tokens = str(query_text).split()
    scores = model.get_scores(q_tokens)
    order = np.argsort(scores)[::-1]

    results = []
    for i in order:
        if exclude_id is not None and doc_ids[i] == exclude_id:
            continue
        results.append((doc_ids[i], float(scores[i])))
        if len(results) == top_n:
            break
    return results

