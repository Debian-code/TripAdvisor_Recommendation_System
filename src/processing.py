import pandas as pd
import typing 
import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from rank_bm25 import BM25Okapi

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_reviews(review : str)->str :
    review=review.lower()
    review=unicodedata.normalize('NFD', review)
    review = review.encode("ascii", "ignore").decode("utf-8")#delete accents    
    review = re.sub(r"[^a-z0-9\s]", " ", review)#delete special caracters
    review = re.sub(r"\s+", " ", review).strip()#delete extra spaces
    return review

def tokenize(review: str) -> list[str]:
    review = review.lower()
    review = unicodedata.normalize('NFD', review)
    review = review.encode("ascii", "ignore").decode("utf-8")
    review = re.sub(r"[^a-z0-9\s]", " ", review)
    tokens = re.findall(r"[a-z0-9]+", review)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def preprocess_to_string(review: str) -> str:
    return " ".join(tokenize(review))


def get_corpus_string(reviews : pd.DataFrame) -> list[str]:
    return reviews["review"].apply(preprocess_to_string).tolist()

def get_corpus_tokens(reviews : pd.DataFrame) -> list[list[str]]:
    return reviews["review"].apply(tokenize).tolist()



def build_place_docs(reviews_df: pd.DataFrame,
                     place_id_col: str = "idplace",
                     text_col: str = "review",
                     lang_col: str = "langue",
                     keep_only_english: bool = True) -> pd.DataFrame:
    """
    Build one document per place by concatenating all (preprocessed) reviews.

    Input: reviews_df with columns: idplace, review, langue (optional)
    Output: DataFrame with columns: idplace, place_document
    """
    df = reviews_df.copy()

    # Optional: keep only English reviews
    if keep_only_english and lang_col in df.columns:
        df = df[df[lang_col].astype(str).str.lower().eq("en")]

    # Drop missing
    df = df.dropna(subset=[place_id_col, text_col])

    # Preprocess each review into a cleaned string
    df["_clean"] = df[text_col].apply(preprocess_to_string)

    # Concatenate by place
    place_docs = (
        df.groupby(place_id_col)["_clean"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index()
        .rename(columns={place_id_col: "idplace", "_clean": "place_document"})
    )

    return place_docs
def get_place_corpus_tokens(place_docs: pd.DataFrame) -> list[list[str]]:
    return place_docs["place_document"].apply(lambda x: str(x).split()).tolist()

def get_place_corpus_string(place_docs: pd.DataFrame) -> list[str]:
    return place_docs["place_document"].fillna("").astype(str).tolist()

def processing_data(reviews : pd.DataFrame,tripadvisor : pd.DataFrame) -> pd.DataFrame:
    first_join=reviews.merge(tripadvisor,left_on="idplace",right_on="id",how="outer")[["idplace","review","langue","typeR","activiteSubCategorie","activiteSubType","restaurantTypeCuisine","priceRange"]]
    first_join=first_join[first_join["langue"]=="en"]
    first_join.drop("langue",axis=1,inplace=True)
    return first_join

def bm25_model(reviews : pd.DataFrame) -> BM25Okapi:
    corpus = get_corpus_tokens(reviews)
    bm25 = BM25Okapi(corpus)
    return bm25

def bm25_score(query: str, bm25_model: BM25Okapi) -> float:
    scores=bm25_model.get_scores(tokenize(query))
    return scores

