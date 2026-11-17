"""Task 3: compare classifiers on the Task 2 DSP features (140k Real and Fake Faces).

Reads the feature arrays saved by ``src.task2`` (no image is decoded again) and fits
Logistic Regression, linear and RBF SVMs, K-Nearest Neighbors, Gaussian Naive Bayes,
and a Decision Tree on the same feature sets. Every hyperparameter is chosen by
5-fold cross-validation on the training shard only; the test shard is used once.
"""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .face_analysis import plot_classifier_comparison
from .task2 import GROUP_NAMES, feature_matrix

# Feature sets compared in Task 3 (keys of task2.EVAL_GROUPS).
FEATURE_SETS = ("band", "radial_hann", "combined")
# Sets wider than this are reduced to PCA_COMPONENTS (fitted on training folds only).
PCA_THRESHOLD = 200
PCA_COMPONENTS = 100
CLASSIFIER_NAMES = {
    "lr": "Logistic Regression",
    "svm_linear": "Linear SVM",
    "svm_rbf": "RBF SVM",
    "knn": "K-Nearest Neighbors",
    "nb": "Gaussian Naive Bayes",
    "tree": "Decision Tree",
}


def classifier_and_grid(name: str, n_inputs: int, seed: int):
    """Return an unfitted classifier and its hyperparameter grid."""
    if name == "lr":
        return (LogisticRegression(max_iter=5000),
                {"C": np.logspace(-4, 3, 8)})
    if name == "svm_linear":
        return (LinearSVC(max_iter=20000, random_state=seed),
                {"C": np.logspace(-4, 0, 5)})
    if name == "svm_rbf":
        # gamma relative to 1/n_inputs, the "scale" default for standardized data.
        return (SVC(kernel="rbf"),
                {"C": [0.1, 1.0, 10.0, 100.0],
                 "gamma": [g / n_inputs for g in (0.01, 0.1, 1.0, 10.0)]})
    if name == "knn":
        return (KNeighborsClassifier(),
                {"n_neighbors": [5, 15, 45, 135]})
    if name == "nb":
        return GaussianNB(), {"var_smoothing": [1e-9]}
    if name == "tree":
        return (DecisionTreeClassifier(random_state=seed),
                {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 20, 50, 100, 200]})
    raise ValueError(f"Unknown classifier {name!r}")


def make_search(name: str, n_features: int, seed: int) -> GridSearchCV:
    """Scaler (+ PCA for wide sets) + classifier, tuned by stratified 5-fold CV."""
    steps = [("scale", StandardScaler())]
    n_inputs = n_features
    if n_features > PCA_THRESHOLD:
        steps.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=seed)))
        n_inputs = PCA_COMPONENTS
    classifier, grid = classifier_and_grid(name, n_inputs, seed)
    steps.append(("clf", classifier))
    return GridSearchCV(Pipeline(steps), {f"clf__{k}": v for k, v in grid.items()},
                        scoring="accuracy", n_jobs=-1, refit=True,
                        cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def scores(model, x: np.ndarray) -> np.ndarray:
    """Continuous FAKE score for ROC AUC (probability or signed margin)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def evaluate(train: dict, test: dict, feature_set: str, name: str, seed: int):
    x_train, x_test = feature_matrix(train, feature_set), feature_matrix(test, feature_set)
    y_train, y_test = train["labels"], test["labels"]
    search = make_search(name, x_train.shape[1], seed)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        search.fit(x_train, y_train)
    search_seconds = time.perf_counter() - start
    model = search.best_estimator_
    predicted = model.predict(x_test)
    confusion = confusion_matrix(y_test, predicted, labels=[0, 1])
    (tn, fp), (fn, tp) = confusion
    params = {key.removeprefix("clf__"): (value.item() if hasattr(value, "item") else value)
              for key, value in search.best_params_.items()}
    row = {
        "feature_set": GROUP_NAMES[feature_set], "set_key": feature_set,
        "classifier": CLASSIFIER_NAMES[name], "clf_key": name,
        "n_features": x_train.shape[1], "n_inputs": model[-1].n_features_in_,
        "best_params": json.dumps(params),
        "cv_accuracy": float(search.best_score_),
        "train_accuracy": float(accuracy_score(y_train, model.predict(x_train))),
        "test_accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores(model, x_test))),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "search_seconds": round(search_seconds, 2),
        "refit_seconds": round(float(search.refit_time_), 3),
    }
    return row, confusion


def load_features(features_dir: Path) -> tuple[dict, dict]:
    features_dir = Path(features_dir)
    splits = []
    for split in ("train", "test"):
        path = features_dir / f"{split}_features.npz"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run src.task2 first.")
        with np.load(path) as stored:
            splits.append({key: stored[key] for key in stored.files})
    return splits[0], splits[1]


def run(features_dir: Path, output_dir: Path, seed: int = 42,
        feature_sets=FEATURE_SETS, classifiers=tuple(CLASSIFIER_NAMES)) -> dict:
    train, test = load_features(features_dir)
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    print(f"train: {len(train['labels'])} images, test: {len(test['labels'])} images")
    metrics, confusions = [], {}
    for feature_set in feature_sets:
        for name in classifiers:
            row, confusion = evaluate(train, test, feature_set, name, seed)
            metrics.append(row)
            confusions[(feature_set, name)] = confusion
            print(f"{row['feature_set']:<30} {row['classifier']:<22} "
                  f"test accuracy {row['test_accuracy']:.4f} (AUC {row['roc_auc']:.4f}, "
                  f"CV {row['cv_accuracy']:.4f}, {row['search_seconds']:.0f} s) "
                  f"{row['best_params']}")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    best = max(metrics, key=lambda row: row["test_accuracy"])
    plot_classifier_comparison(metrics, confusions[(best["set_key"], best["clf_key"])],
                               f"{best['classifier']}, {best['feature_set']}",
                               output_dir / "figures" / "classifiers.png")
    metadata = {
        "features_dir": str(Path(features_dir)), "seed": seed,
        "counts": {"train": int(len(train["labels"])), "test": int(len(test["labels"]))},
        "feature_sets": {key: GROUP_NAMES[key] for key in feature_sets},
        "preprocessing": (f"StandardScaler; sets with more than {PCA_THRESHOLD} features "
                          f"then PCA to {PCA_COMPONENTS} components (fit on training folds)"),
        "model_selection": "GridSearchCV, accuracy, StratifiedKFold(5, shuffle) on training only",
        "grid_note": "svm_rbf gamma values are multiples of 1 / (number of model inputs)",
        "grids": {name: {k: [float(v) if isinstance(v, (int, float, np.floating)) else v
                             for v in values]
                         for k, values in classifier_and_grid(name, 1, seed)[1].items()}
                  for name in classifiers},
        "versions": {name: version(name) for name in ("numpy", "scikit-learn")},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"Saved metrics, figure, and metadata to {output_dir.resolve()}")
    return {"metrics": metrics, "metadata": metadata}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/faces/features"),
                        help="Directory with train_features.npz and test_features.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        run(args.features_dir, args.output_dir, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
