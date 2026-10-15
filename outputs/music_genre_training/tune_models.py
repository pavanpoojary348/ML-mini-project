"""Compare stronger FMA Small candidates on the official validation split only."""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

from train_fma import DEFAULT_AUDIO, DEFAULT_METADATA, read_labels, read_target_features, build_model


def candidates():
    return {
        "Baseline_RandomForest": build_model("RandomForest"),
        "Baseline_ExtraTrees": build_model("ExtraTrees"),
        "SVM_C3": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=3, kernel="rbf", class_weight="balanced", cache_size=512)),
        "SVM_Select128_C3": make_pipeline(
            SimpleImputer(strategy="median"), SelectKBest(f_classif, k=128), StandardScaler(),
            SVC(C=3, kernel="rbf", class_weight="balanced", cache_size=512),
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.08, max_leaf_nodes=20, l2_regularization=2.0, random_state=42),
        ),
        "MLP_256_128": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=0.001, batch_size=128, learning_rate_init=0.001, max_iter=80, early_stopping=False, random_state=42),
        ),
    }


def main():
    root = Path(__file__).resolve().parent
    metadata_dir, audio_dir = DEFAULT_METADATA, DEFAULT_AUDIO
    for path in (metadata_dir / "tracks.csv", metadata_dir / "features.csv", audio_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    labels = read_labels(metadata_dir / "tracks.csv", chunk_size=5000)
    small = labels.loc[labels["subset"].eq("small")]
    train = small.loc[small["split"].eq("training")]
    valid = small.loc[small["split"].eq("validation")]
    features = read_target_features(metadata_dir / "features.csv", set(train.index) | set(valid.index), chunk_size=5000)
    train_ids, valid_ids = train.index.intersection(features.index), valid.index.intersection(features.index)
    X_train = features.loc[train_ids].to_numpy(dtype=np.float32)
    y_train = train.loc[train_ids, "genre"].astype(str).to_numpy()
    X_valid = features.loc[valid_ids].to_numpy(dtype=np.float32)
    y_valid = valid.loc[valid_ids, "genre"].astype(str).to_numpy()
    print(f"Training only on {len(train_ids)} tracks; validating on {len(valid_ids)} tracks.", flush=True)

    rows = []
    best_name, best_score, best_model = None, (-1.0, -1.0), None
    for name, model in candidates().items():
        print(f"Fitting {name}...", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_train, y_train)
            prediction = model.predict(X_valid)
            accuracy = float(accuracy_score(y_valid, prediction))
            macro_f1 = float(f1_score(y_valid, prediction, average="macro"))
            rows.append({"model": name, "validation_accuracy": accuracy, "validation_macro_f1": macro_f1})
            print(f"  accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}", flush=True)
        except Exception as exc:
            rows.append({"model": name, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
            continue
        score = (macro_f1, accuracy)
        if score > best_score:
            best_score, best_name, best_model = score, name, model

    results = pd.DataFrame(rows).sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False)
    out = root / "tuning_results"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "validation_leaderboard.csv", index=False)
    joblib.dump(
        {"model": best_model, "feature_columns": list(features.columns), "classes": sorted(np.unique(y_train)), "name": best_name},
        out / "best_validation_model.joblib",
        compress=3,
    )
    print("\nVALIDATION LEADERBOARD", flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"\nBest validation model: {best_name}. No test tracks were used.", flush=True)
    print(f"Saved tuning artifacts to {out}", flush=True)


if __name__ == "__main__":
    main()
