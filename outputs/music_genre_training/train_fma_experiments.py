"""FMA Small feature analysis, tuned gradient boosting, and validation error study.

All model selection uses the official validation set. The test set is held out
until the final chosen configuration and augmentation policy have been fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from outputs.music_genre_training.train_fma import (  # noqa: E402
    DEFAULT_AUDIO, DEFAULT_METADATA, read_labels, read_target_features,
    build_model,
)

SEED = 42


def load_data(metadata_dir: Path, chunk_size: int = 5000):
    labels = read_labels(metadata_dir / "tracks.csv", chunk_size)
    small = labels.loc[labels.subset.eq("small")]
    tr = small.loc[small.split.eq("training")]
    va = small.loc[small.split.eq("validation")]
    te = small.loc[small.split.eq("test")]
    feats = read_target_features(metadata_dir / "features.csv", set(tr.index)|set(va.index)|set(te.index), chunk_size)
    ti, vi, ei = tr.index.intersection(feats.index), va.index.intersection(feats.index), te.index.intersection(feats.index)
    return feats, tr.loc[ti], va.loc[vi], te.loc[ei]


def engineer(features: pd.DataFrame) -> pd.DataFrame:
    """Keep original descriptors and add robust family-level summary features."""
    out = features.copy()
    groups: dict[str, list[str]] = {}
    for col in features.columns:
        family = col.split("__", 1)[0]
        groups.setdefault(family, []).append(col)
    for family, cols in groups.items():
        block = features[cols].apply(pd.to_numeric, errors="coerce")
        out[f"engineered__{family}__row_mean"] = block.mean(axis=1)
        out[f"engineered__{family}__row_std"] = block.std(axis=1)
        out[f"engineered__{family}__row_range"] = block.max(axis=1) - block.min(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}


def analyze(feats, train, valid, test, out):
    target = out / "dataset_analysis.json"
    summary = {
        "dataset": "FMA Small", "source": "tracks.csv + features.csv",
        "official_split_tracks": {"training_labels": int(len(train)), "validation_labels": int(len(valid)), "test_labels": int(len(test))},
        "feature_rows_matched": int(len(feats)), "feature_count": int(feats.shape[1]),
        "missing_feature_values": int(feats.isna().sum().sum()),
        "missing_fraction": float(feats.isna().to_numpy().mean()),
        "class_counts": {k: int(v) for k,v in train.genre.value_counts().sort_index().items()},
        "validation_class_counts": {k: int(v) for k,v in valid.genre.value_counts().sort_index().items()},
        "feature_families": sorted({str(c).split("__",1)[0] for c in feats.columns}),
        "target_notes": ["Eight balanced target genres; genre labels are subjective and acoustically overlapping.",
                         "FMA Small has about 8,000 tracks; missing published feature rows reduce available tracks.",
                         "Use official validation/test splits; do not use test outcomes for feature or hyperparameter selection."],
    }
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_baseline(feats, train, valid, out):
    tr_ids, va_ids = train.index, valid.index
    Xtr, Xva = feats.loc[tr_ids].to_numpy(np.float32), feats.loc[va_ids].to_numpy(np.float32)
    ytr, yva = train.genre.astype(str).to_numpy(), valid.genre.astype(str).to_numpy()
    model = build_model("SVM_C3")
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    result = {"model": "RBF SVM C=3", "features": int(feats.shape[1]), "train_tracks": len(ytr), "validation_tracks": len(yva), **metrics(yva,pred),
              "baseline_reasoning": ["518 aggregate descriptors compress away local temporal/rhythmic patterns.",
                                     "Genre boundaries overlap, especially Folk, Experimental, Instrumental and Pop.",
                                     "The official split is track-disjoint and harder than potentially ungrouped random image splits.",
                                     "Only 6,339 training rows with most rows sharing similar engineered summary statistics."]}
    (out/"baseline_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


def tune(feats, train, valid, out, n_trials):
    # Optuna is vendored under work/optuna_pkg so this project does not need to
    # modify the user's global Python installation.
    vendored = ROOT / "work" / "optuna_pkg"
    if vendored.exists():
        sys.path.insert(0, str(vendored))
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna missing. Install with: python -m pip install optuna") from exc
    Xtr = feats.loc[train.index].to_numpy(np.float32)
    Xva = feats.loc[valid.index].to_numpy(np.float32)
    ytr, yva = train.genre.astype(str).to_numpy(), valid.genre.astype(str).to_numpy()
    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 250, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80, step=5),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 30, log=True),
            "max_bins": trial.suggest_int("max_bins", 64, 160, step=32),
            "random_state": SEED, "early_stopping": True, "validation_fraction": 0.12,
        }
        model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(**params))
        model.fit(Xtr,ytr)
        pred = model.predict(Xva)
        score = metrics(yva,pred)
        trial.set_user_attr("validation_accuracy", score["accuracy"])
        trial.set_user_attr("validation_macro_f1", score["macro_f1"])
        return score["macro_f1"]
    study = optuna.create_study(direction="maximize", study_name="fma_small_histgb_macro_f1", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    trials = study.trials_dataframe(attrs=("number","value","params","user_attrs","state"))
    trials.to_csv(out/"optuna_trials.csv",index=False)
    best = {"best_validation_macro_f1": study.best_value, "best_params": study.best_params, "trial_count": len(study.trials),
            "objective": "validation macro-F1", "test_used_for_tuning": False}
    (out/"optuna_best.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
    print(json.dumps(best,indent=2))


def make_hgb(params, sample_weight=None):
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(**params, random_state=SEED, early_stopping=True, validation_fraction=.12))


def error_analysis(feats, train, valid, out, best_params, aug_factor=1.0):
    Xtr=feats.loc[train.index].to_numpy(np.float32); Xva=feats.loc[valid.index].to_numpy(np.float32)
    ytr=train.genre.astype(str).to_numpy(); yva=valid.genre.astype(str).to_numpy()
    params=dict(best_params)
    base=make_hgb(params); base.fit(Xtr,ytr); pred=base.predict(Xva)
    labels=sorted(np.unique(ytr))
    cm=confusion_matrix(yva,pred,labels=labels)
    per=classification_report(yva,pred,labels=labels,output_dict=True,zero_division=0)
    analysis=[]
    for i,c in enumerate(labels):
        row={"genre":c,"support":int(cm[i].sum()),"recall":float(per[c]["recall"]),"f1":float(per[c]["f1-score"]),"correct":int(cm[i,i]),"errors":int(cm[i].sum()-cm[i,i])}
        row["most_confused_as"] = labels[int(np.argmax(np.where(np.arange(len(labels))==i,-1,cm[i])))]
        analysis.append(row)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/"validation_confusion_matrix.csv")
    pd.DataFrame(analysis).sort_values("recall").to_csv(out/"validation_class_errors.csv",index=False)
    pd.DataFrame({"track_id":valid.index.to_numpy(),"true_genre":yva,"predicted_genre":pred,"correct":pred==yva}).to_csv(out/"validation_misclassifications.csv",index=False)
    base_score=metrics(yva,pred)
    # Targeted tabular augmentation: duplicate weak-class train examples with small
    # Gaussian perturbations, scaled per feature; only classes underperforming the
    # validation average are augmented. Never augment validation/test rows.
    weak=[r["genre"] for r in analysis if r["recall"] < base_score["accuracy"]]
    mu=np.nanmedian(Xtr,axis=0); scale=np.nanmedian(np.abs(Xtr-mu),axis=0)*1.4826
    scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.0)
    rng=np.random.default_rng(SEED)
    synth=[]; synth_y=[]
    for cls in weak:
        idx=np.flatnonzero(ytr==cls)
        n=max(1,int(len(idx)*aug_factor))
        choose=rng.choice(idx,size=n,replace=True)
        noise=rng.normal(0,0.025,size=(n,Xtr.shape[1]))*scale
        synth.append(Xtr[choose]+noise); synth_y.extend([cls]*n)
    X_aug=np.concatenate([Xtr,*synth]) if synth else Xtr
    y_aug=np.concatenate([ytr,np.asarray(synth_y)]) if synth else ytr
    augmented=make_hgb(params); augmented.fit(X_aug,y_aug); pred2=augmented.predict(Xva)
    aug_score=metrics(yva,pred2)
    out_result={"best_params":params,"weak_validation_classes_augmented":weak,"augmentation":"train-only Gaussian jittered replicas at 2.5% robust feature scale",
       "synthetic_rows_added":len(y_aug)-len(ytr),"baseline_validation":base_score,"augmented_validation":aug_score,
       "augmentation_helped":bool(aug_score["macro_f1"]>base_score["macro_f1"]),"class_errors":analysis,
       "interpretation":"Augmentation selected by validation recall; validation examples were not included in augmented training."}
    (out/"validation_error_analysis.json").write_text(json.dumps(out_result,indent=2),encoding="utf-8")
    print(json.dumps(out_result,indent=2))
    return augmented, aug_score, base_score, weak


def final_test(feats, train, valid, test, out, params, weak, aug_factor=1.0):
    fit=pd.concat([train,valid]); X=feats.loc[fit.index].to_numpy(np.float32); y=fit.genre.astype(str).to_numpy()
    Xt=feats.loc[test.index].to_numpy(np.float32); yt=test.genre.astype(str).to_numpy()
    mu=np.nanmedian(X,axis=0); scale=np.nanmedian(np.abs(X-mu),axis=0)*1.4826; scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.)
    rng=np.random.default_rng(SEED); synth=[]; sy=[]
    for cls in weak:
        idx=np.flatnonzero(y==cls)
        if not len(idx): continue
        n=max(1,int(len(idx)*aug_factor)); choose=rng.choice(idx,n,replace=True)
        synth.append(X[choose]+rng.normal(0,.025,size=(n,X.shape[1]))*scale); sy.extend([cls]*n)
    if synth: X=np.concatenate([X,*synth]); y=np.concatenate([y,np.asarray(sy)])
    model=make_hgb(params); model.fit(X,y); pred=model.predict(Xt); score=metrics(yt,pred)
    result={"dataset":"FMA Small","test_tracks":len(yt),"feature_count":int(feats.shape[1]),"model":"HistGradientBoosting with feature engineering + targeted train-only jitter",
       "test_accuracy":score["accuracy"],"test_macro_f1":score["macro_f1"],"classification_report":classification_report(yt,pred,output_dict=True,zero_division=0),"test_used_for_selection":False,"params":params,"augmented_classes":weak}
    (out/"final_test_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame(confusion_matrix(yt,pred,labels=sorted(np.unique(yt))),index=sorted(np.unique(yt)),columns=sorted(np.unique(yt))).to_csv(out/"final_test_confusion_matrix.csv")
    joblib.dump({"model":model,"feature_columns":list(feats.columns),"classes":sorted(np.unique(yt))},out/"feature_engineered_hgb.joblib",compress=3)
    print(json.dumps({k:v for k,v in result.items() if k!="classification_report"},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-dir",type=Path,default=DEFAULT_METADATA)
    p.add_argument("--audio-dir",type=Path,default=DEFAULT_AUDIO)
    p.add_argument("--output-dir",type=Path,default=HERE/"advanced_results")
    p.add_argument("--trials",type=int,default=50)
    p.add_argument("--augmentation-factor",type=float,default=1.0)
    p.add_argument("--skip-tuning",action="store_true")
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    for x in (a.metadata_dir/"tracks.csv",a.metadata_dir/"features.csv",a.audio_dir):
        if not x.exists(): raise FileNotFoundError(x)
    print("Loading official FMA Small splits and precomputed features",flush=True)
    raw,train,valid,test=load_data(a.metadata_dir)
    analyze(raw,train,valid,test,a.output_dir)
    print("Baseline",flush=True); run_baseline(raw,train,valid,a.output_dir)
    feats=engineer(raw)
    (a.output_dir/"engineered_feature_manifest.json").write_text(json.dumps({"original_features":raw.shape[1],"engineered_features":feats.shape[1]-raw.shape[1],"total_features":feats.shape[1],"families":sorted({c.split("__",1)[0] for c in raw.columns})},indent=2),encoding="utf-8")
    if not a.skip_tuning: tune(feats,train,valid,a.output_dir,a.trials)
    tuned=json.loads((a.output_dir/"optuna_best.json").read_text(encoding="utf-8"))
    _,aug_score,base_score,weak=error_analysis(feats,train,valid,a.output_dir,tuned["best_params"],a.augmentation_factor)
    # Prefer targeted augmentation only if it improves validation macro-F1.
    if aug_score["macro_f1"] <= base_score["macro_f1"]: weak=[]
    final_test(feats,train,valid,test,a.output_dir,tuned["best_params"],weak,a.augmentation_factor)

if __name__=="__main__": main()
