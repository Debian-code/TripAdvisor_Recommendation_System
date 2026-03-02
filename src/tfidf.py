# tfidf.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_fit(place_docs_df, max_features=20000, ngram_range=(1,2), min_df=2):
    doc_ids = place_docs_df["idplace"].tolist()
    texts = place_docs_df["place_document"].fillna("").tolist()
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(texts)
    return vec, X, doc_ids

def tfidf_rank(vec, X, doc_ids, query_text: str, top_n=100, exclude_id=None):
    q = vec.transform([str(query_text)])
    sims = cosine_similarity(q, X).ravel()
    order = np.argsort(sims)[::-1]

    results = []
    for i in order:
        if exclude_id is not None and doc_ids[i] == exclude_id:
            continue
        results.append((doc_ids[i], float(sims[i])))
        if len(results) == top_n:
            break
    return results