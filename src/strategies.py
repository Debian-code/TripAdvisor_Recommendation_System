# strategies.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[:max_words])

def apply_strategy_truncate(place_docs: pd.DataFrame, max_words=1200) -> pd.DataFrame:
    out = place_docs.copy()
    out["place_document"] = out["place_document"].fillna("").apply(lambda t: truncate_words(t, max_words))
    return out

def apply_strategy_topk_tfidf_words(place_docs: pd.DataFrame, k_words=200, max_features=50000, min_df=2) -> pd.DataFrame:
    texts = place_docs["place_document"].fillna("").tolist()
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    reduced = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            reduced.append("")
            continue
        idx = row.indices
        data = row.data
        top = idx[np.argsort(data)[::-1]][:k_words]
        reduced.append(" ".join(vocab[top].tolist()))

    out = place_docs.copy()
    out["place_document"] = reduced
    return out