from pathlib import Path

import joblib

MODEL_PATH = Path("models/request_router.joblib")


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Train the model first with src/train.py.")
    return joblib.load(path)


def predict(text, model=None):
    model = model or load_model()
    return str(model.predict([text])[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(predict(args.text))
