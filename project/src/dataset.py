
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataframe(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

   
    df["text"] = df["lyrics_bn"].astype(str) + " " + df["english lyrics"].astype(str)


    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def make_tfidf(df: pd.DataFrame, max_features: int = 2000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(df["text"]).toarray().astype("float32")
    return X, vectorizer
