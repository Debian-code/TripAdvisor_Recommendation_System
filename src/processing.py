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




def processing_data(reviews : pd.DataFrame,tripadvisor : pd.DataFrame) -> pd.DataFrame:
    first_join=reviews.merge(tripadvisor,left_on="idplace",right_on="id",how="outer")[["idplace","nom","review","langue","typeR","activiteSubCategorie","activiteSubType","restaurantTypeCuisine","priceRange"]]
    first_join=first_join[first_join["langue"]=="en"]
    first_join.drop("langue",axis=1,inplace=True)
    return first_join



