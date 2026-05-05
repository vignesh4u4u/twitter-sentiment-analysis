from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


DATA_PATH = Path("twitter.csv")
MODEL_PATH = Path("sentiment_model.pkl")
RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 35000
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def load_data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=["clean_text", "category"])
    data["clean_text"] = data["clean_text"].astype(str).str.strip()
    data["category"] = pd.to_numeric(data["category"], errors="coerce")
    data = data[data["category"].isin([-1, 0, 1])]
    data = data[data["clean_text"].ne("")]
    data["category"] = data["category"].astype(int)
    return data


def equal_sample(data):
    counts = data["category"].value_counts()
    sample_size = min(SAMPLE_SIZE_PER_CLASS, int(counts.min()))
    sampled = [
        data[data["category"].eq(label)].sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        for label in [-1, 0, 1]
    ]
    data = pd.concat(sampled, ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return data, sample_size


def build_model():
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=250000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=150000,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=1000,
                    solver="saga",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    data = load_data(DATA_PATH)
    original_counts = data["category"].value_counts().sort_index()
    data, sample_size = equal_sample(data)
    x = data["clean_text"]
    y = data["category"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[-1, 0, 1],
        target_names=[LABEL_NAMES[-1], LABEL_NAMES[0], LABEL_NAMES[1]],
        digits=4,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[-1, 0, 1])

    joblib.dump(
        {
            "model": model,
            "label_names": LABEL_NAMES,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "sample_size_per_class": sample_size,
        },
        MODEL_PATH,
    )

    print(f"clean rows: {int(original_counts.sum())}")
    print("original label counts:")
    print(original_counts.to_string())
    print(f"sample size per class: {sample_size}")
    print(f"training rows after equal sampling: {len(data)}")
    print("sampled label counts:")
    print(y.value_counts().sort_index().to_string())
    print(f"accuracy: {accuracy:.4f}")
    print(report)
    print("confusion matrix labels: -1, 0, 1")
    print(matrix)
    print(f"saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
