# FMA Small music genre model training

This starter training project uses FMA's precomputed audio features. By default, it trains only on the FMA Small training split, compares Extra Trees, Random Forest, an RBF-SVM, and HistGradientBoosting, selects using validation macro-F1, refits the selected model on training plus validation data, and evaluates it on the separate FMA Small test split. An optional experiment can include additional medium/large tracks, but that lowered held-out accuracy in our first comparison.

The feature-based algorithms are a different approach from the reference paper's CNN and MobileNetV2 models. This first run focuses on establishing an honest baseline; a higher accuracy is a goal, not something we can guarantee in advance.

## Expected data folders

The script uses the extracted folders in Downloads by default:

- Audio: `C:\Users\pooja\Downloads\fma_small\fma_small`
- Metadata: `C:\Users\pooja\Downloads\fma_metadata\fma_metadata`

The metadata folder must contain `tracks.csv` and `features.csv`. The audio folder is checked so the script can confirm the dataset location.

## Run

Open PowerShell in this folder and run:

```powershell
python train_fma.py
```

The script scans the large feature CSV in chunks, so it may take several minutes. It needs pandas, NumPy, scikit-learn, matplotlib, and joblib. Results are saved in `results/`. Validation-only tuning results are saved in `tuning_results/`; no test tracks were used to choose the model. The earlier small-only baseline and larger-data comparison are also preserved in their result folders.

- `metrics.json`: validation comparison and held-out test metrics
- `confusion_matrix.png` and `.csv`: test-set errors by genre
- `best_model.joblib`: selected trained model and feature names

To override the default data paths:

```powershell
python train_fma.py --metadata-dir "C:\path\to\fma_metadata" --audio-dir "C:\path\to\fma_small"
```

To repeat the larger-data comparison, use a separate output folder:

```powershell
python train_fma.py --include-extra-data --output-dir results_with_extra_data
```

To explicitly fit one of the validation candidates, pass `--force-model` (for example, `--force-model SVM_C3`).

## Pretrained BYOL-A experiment

The BYOL-A experiment (`train_byola.py`) uses an audio encoder pretrained with self-supervised learning, then trains an MLP genre classifier on its song-level audio embeddings. A 2026 study reports 64.3% FMA-Small accuracy for BYOL-A embeddings with a deep classifier; that published figure is a reference, not a promise, since exact preprocessing and evaluation details may differ. This project uses FMA's official train, validation, and test split, selects the epoch on validation macro-F1, then refits on train plus validation and evaluates the test split once.

The BYOL-A code and weights are supplied under the authors' evaluation license. Do not redistribute the pretrained weights; review the license before use beyond a classroom evaluation. To prepare the local reference files:

```powershell
git clone --depth 1 https://github.com/nttcslab/byol-a.git work\byol-a
```

Install `audio_requirements.txt`, then run from this folder:

```powershell
python train_byola.py
```

The first run extracts six audio crops for each track and saves a resumable cache in `byola_results/cache/`. Results include `metrics.json`, `confusion_matrix.csv`, `training_history.csv`, and the fitted MLP checkpoint. The pretrained MERT experiment is also available in `train_mert_embeddings.py` for an alternative comparison.

Research references: [BYOL-A genre-classification study](https://arxiv.org/abs/2603.13871), [official BYOL-A repository](https://github.com/nttcslab/byol-a), and [MERT paper](https://arxiv.org/abs/2306.00107).

## Feature engineering and Gradient Boosting experiment

`train_fma_experiments.py` analyzes the FMA Small splits, records the RBF-SVM baseline, adds 33 family-level summary features to the 518 published descriptors, runs 50 Optuna trials for HistGradientBoosting on validation macro-F1, studies validation errors, and tries small train-only augmentation for the weakest validation classes. The official test split is used once at the end.

Install Optuna in the Python environment, then run:

```powershell
python -m pip install optuna
python train_fma_experiments.py --trials 50
```

Results, including the full trial leaderboard and the final report, are written to `advanced_results/`. In this run, Gradient Boosting reached 58.9% validation accuracy, but 47.1% test accuracy after the augmentation chosen by validation macro-F1. BYOL-A + MLP remains the strongest measured test model at 53.0%. See [`advanced_results/training_report.md`](advanced_results/training_report.md) for the results and error analysis. These results do not support claiming 80% or 90% accuracy on this official FMA Small evaluation.
