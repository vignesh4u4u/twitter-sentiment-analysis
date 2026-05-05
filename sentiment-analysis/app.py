from pathlib import Path

import joblib


MODEL_PATH = Path("sentiment_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("sentiment_model.pkl not found. Run python model_train.py first.")
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["label_names"]


def predict_sentiment(model, label_names, text):
    prediction = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    classes = [int(label) for label in model.classes_]
    scores = dict(zip(classes, probabilities))
    confidence = float(scores[prediction])
    return prediction, label_names[prediction], confidence, scores


def format_scores(label_names, scores):
    return " | ".join(
        f"{label_names[label]} {scores[label] * 100:.2f}%"
        for label in [-1, 0, 1]
    )


def main():
    model, label_names = load_model()
    print("Sentiment Analysis")
    print("Enter text and press Enter. Type exit to stop.")

    while True:
        text = input("Text: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            print("Please enter text.")
            continue
        label, sentiment, confidence, scores = predict_sentiment(model, label_names, text)
        print(f"Label: {label}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Scores: {format_scores(label_names, scores)}")


if __name__ == "__main__":
    main()
