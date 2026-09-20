import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_models():
    common = dict(
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**common)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**common)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    return predictions, report, metrics


def main(csv_path, output_dir="results", model_dir="models"):
    df = pd.read_csv(csv_path)
    required = {"Document", "Topic_group"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing text or labels.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["Document"],
        df["Topic_group"],
        test_size=0.20,
        random_state=42,
        stratify=df["Topic_group"],
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "records": int(len(df)),
        "classes": int(df["Topic_group"].nunique()),
        "train_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_documents": int(df["Document"].duplicated().sum()),
        "models": {},
    }

    best_name = None
    best_macro_f1 = -1
    best_model = None

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions, report, metrics = evaluate(model, x_test, y_test)
        all_metrics["models"][name] = metrics

        pd.DataFrame(
            {
                "text": x_test.values,
                "actual": y_test.values,
                "predicted": predictions,
                "correct": y_test.values == predictions,
            }
        ).to_csv(Path(output_dir) / f"{name}_predictions.csv", index=False)

        pd.DataFrame(report).T.to_csv(Path(output_dir) / f"{name}_classification_report.csv")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_name = name
            best_model = model

    all_metrics["best_model"] = best_name
    all_metrics["best_macro_f1"] = best_macro_f1
    Path(output_dir, "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    joblib.dump(best_model, Path(model_dir, "request_router.joblib"))
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    main(args.csv_path, args.output_dir, args.model_dir)
