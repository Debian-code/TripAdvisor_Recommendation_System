from rank_bm25 import BM25Okapi 
import processing
import pandas as pd


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

