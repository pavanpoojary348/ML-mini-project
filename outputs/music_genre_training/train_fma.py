"""Train and evaluate music-genre classifiers on FMA Small features.

This training stage uses FMA's published audio-feature table, so the script
does not need to decode MP3 files or install a deep-learning framework. It
compares tree-based ML algorithms, training on additional labeled FMA tracks
while keeping the FMA Small validation and test tracks separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_ROOT = Path.home() / "Downloads"
DEFAULT_METADATA = DEFAULT_ROOT / "fma_metadata" / "fma_metadata"
DEFAULT_AUDIO = DEFAULT_ROOT / "fma_small" / "fma_small"


def read_labels(tracks_csv: Path, chunk_size: int) -> pd.DataFrame:
    wanted = []
    cols = [("set", "subset"), ("set", "split"), ("track", "genre_top")]
    for chunk in pd.read_csv(
        tracks_csv,
        header=[0, 1],
        index_col=0,
        chunksize=chunk_size,
    ):
        labeled = chunk.loc[
            chunk[("track", "genre_top")].notna()
            & chunk[("set", "subset")].isin(["small", "medium", "large"]),
            cols,
        ]
        if not labeled.empty:
            wanted.append(labeled)
    if not wanted:
        raise RuntimeError("No labeled FMA rows found in tracks.csv")
    labels = pd.concat(wanted)
    labels.columns = [str(c[-1]) for c in labels.columns]
    labels = labels.rename(columns={"genre_top": "genre"})
    labels.index.name = "track_id"
    labels = labels.dropna(subset=["genre", "split", "subset"])
    labels["track_id"] = labels.index.astype(int)
    return labels


def read_target_features(features_csv: Path, track_ids: set[int], chunk_size: int) -> pd.DataFrame:
    selected = []
    total_chunks = 0
    for chunk in pd.read_csv(
        features_csv, header=[0, 1], index_col=0, chunksize=chunk_size
    ):
        total_chunks += 1
        ids = chunk.index.intersection(track_ids)
        if len(ids):
            selected.append(chunk.loc[ids])
        if total_chunks % 5 == 0:
            print(f"Scanned {total_chunks} feature chunks; found {sum(len(x) for x in selected)} target tracks.", flush=True)
    if not selected:
        raise RuntimeError("Could not find FMA Small track IDs in features.csv")
    features = pd.concat(selected).sort_index()
    features = features.apply(pd.to_numeric, errors="coerce")
    features.columns = [
        "__".join(str(part) for part in col if str(part) != "")
        if isinstance(col, tuple)
        else str(col)
        for col in features.columns
    ]
    features.index.name = "track_id"
    return features


def build_model(name: str):
    if name.startswith("SVM_C"):
        c_value = float(name.removeprefix("SVM_C"))
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=c_value, kernel="rbf", class_weight="balanced", cache_size=512),
        )
    if name == "ExtraTrees":
        classifier = ExtraTreesClassifier(
            n_estimators=350,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    elif name == "RandomForest":
        classifier = RandomForestClassifier(
            n_estimators=350,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    elif name == "HistGradientBoosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.08,
                max_leaf_nodes=20,
                l2_regularization=2.0,
                random_state=42,
            ),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(SimpleImputer(strategy="median"), classifier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--include-extra-data",
        action="store_true",
        help="Also train on medium/large labeled tracks (experimental; this lowered test accuracy in our first comparison).",
    )
    parser.add_argument(
        "--force-model",
        choices=["ExtraTrees", "RandomForest", "SVM_C3", "HistGradientBoosting"],
        help="Use a specific candidate after validation scoring instead of selecting the validation winner.",
    )
    args = parser.parse_args()

    tracks_csv = args.metadata_dir / "tracks.csv"
    features_csv = args.metadata_dir / "features.csv"
    for path in (tracks_csv, features_csv, args.audio_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading FMA genre labels and official data splits...", flush=True)
    labels = read_labels(tracks_csv, args.chunk_size)
    small = labels.loc[labels["subset"].eq("small")]
    classes = sorted(small["genre"].unique())
    extra_train = labels.loc[
        labels["subset"].isin(["medium", "large"])
        & labels["split"].eq("training")
        & labels["genre"].isin(classes)
    ]
    if not args.include_extra_data:
        extra_train = extra_train.iloc[0:0]
    train_labels = pd.concat([small.loc[small["split"].eq("training")], extra_train]).sort_index()
    valid_labels = small.loc[small["split"].eq("validation")].sort_index()
    test_labels = small.loc[small["split"].eq("test")].sort_index()
    print(
        f"Training tracks: {len(train_labels)} total (including {len(extra_train)} additional medium/large tracks); "
        f"validation tracks: {len(valid_labels)}; test tracks: {len(test_labels)}.",
        flush=True,
    )
    print("Training label counts:", flush=True)
    print(train_labels["genre"].value_counts().sort_index().to_string(), flush=True)

    target_ids = set(train_labels.index) | set(valid_labels.index) | set(test_labels.index)
    print("Scanning the published audio-feature table for training and FMA Small evaluation tracks...", flush=True)
    features = read_target_features(features_csv, target_ids, args.chunk_size)
    train_ids = train_labels.index.intersection(features.index)
    valid_ids = valid_labels.index.intersection(features.index)
    test_ids = test_labels.index.intersection(features.index)
    if len(test_ids) < 700 or len(valid_ids) < 700:
        raise RuntimeError(f"Too few FMA Small evaluation tracks matched: validation={len(valid_ids)}, test={len(test_ids)}")
    X_train = features.loc[train_ids].to_numpy(dtype=np.float32)
    y_train = train_labels.loc[train_ids, "genre"].astype(str).to_numpy()
    X_valid = features.loc[valid_ids].to_numpy(dtype=np.float32)
    y_valid = valid_labels.loc[valid_ids, "genre"].astype(str).to_numpy()
    X_test = features.loc[test_ids].to_numpy(dtype=np.float32)
    y_test = test_labels.loc[test_ids, "genre"].astype(str).to_numpy()
    print(
        f"Matched {len(train_ids)} training, {len(valid_ids)} validation, and {len(test_ids)} test tracks with {features.shape[1]} features.",
        flush=True,
    )

    models = {}
    validation_scores = {}
    candidate_names = [args.force_model] if args.force_model else ["ExtraTrees", "RandomForest", "SVM_C3", "HistGradientBoosting"]
    for name in candidate_names:
        print(f"Training {name}...", flush=True)
        model = build_model(name)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        validation_scores[name] = {
            "accuracy": float(accuracy_score(y_valid, pred)),
            "macro_f1": float(f1_score(y_valid, pred, average="macro")),
        }
        models[name] = model
        print(f"{name} validation: {validation_scores[name]}", flush=True)

    best_name = args.force_model or max(
        validation_scores, key=lambda n: (validation_scores[n]["macro_f1"], validation_scores[n]["accuracy"])
    )
    best_model = models[best_name]
    print(f"Refitting selected {best_name} on training plus validation tracks...", flush=True)
    best_model.fit(
        np.concatenate([X_train, X_valid]),
        np.concatenate([y_train, y_valid]),
    )
    test_pred = best_model.predict(X_test)
    class_names = sorted(np.unique(y_train))
    test_accuracy = float(accuracy_score(y_test, test_pred))
    test_macro_f1 = float(f1_score(y_test, test_pred, average="macro"))
    report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format="d"
    )
    ax.tick_params(axis="both", labelsize=9)
    plt.title(f"FMA Small test set - {best_name} (accuracy {test_accuracy:.1%})")
    fig.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    result = {
        "dataset": "FMA Small",
        "training_scope": "FMA Small training split plus medium/large tracks" if args.include_extra_data else "FMA Small training split only",
        "training_tracks": int(len(train_ids)),
        "extra_medium_large_training_tracks": int(len(extra_train)),
        "validation_tracks": int(len(valid_ids)),
        "test_tracks": int(len(test_ids)),
        "feature_count": int(features.shape[1]),
        "split_counts": {"training": int(len(train_ids)), "validation": int(len(valid_ids)), "test": int(len(test_ids))},
        "validation_scores": validation_scores,
        "selected_model": best_name,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_classification_report": report,
        "classes": class_names,
        "feature_columns": list(features.columns),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(args.output_dir / "confusion_matrix.csv")
    joblib.dump(
        {"model": best_model, "feature_columns": list(features.columns), "classes": class_names},
        args.output_dir / "best_model.joblib",
        compress=3,
    )
    print("\nFINAL TEST RESULTS", flush=True)
    print(f"Selected model: {best_name}", flush=True)
    print(f"Test accuracy: {test_accuracy:.2%}", flush=True)
    print(f"Test macro-F1: {test_macro_f1:.4f}", flush=True)
    print(f"Results saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
