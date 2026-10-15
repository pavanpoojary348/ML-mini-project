# FMA Small: baseline, boosting, tuning, and error analysis

## Dataset analysis

This experiment uses FMA Small's official training, validation, and test splits. Of 6,339 labeled training tracks, 794 validation tracks, and 785 test tracks, all have a row in the published 518-column feature table. The table covers 11 descriptor families: chroma, MFCC, RMS energy, spectral, tonnetz, and zero-crossing features. The retained feature rows contain no missing values. Training labels are nearly balanced (772–800 tracks per genre), so class imbalance is not the main accuracy bottleneck.

The descriptors are song-level summary statistics. They do not preserve the order of events through a song or all the local rhythm/timbre changes. Genres also overlap: the single-label task forces tracks with mixed styles into one genre.

## Results

| Model | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| RBF SVM on 518 published features | 56.2% | 55.2% | 49.3% | 49.0% |
| Feature engineered HistGradientBoosting, best of 50 trials | 58.9% | 58.5% | — | — |
| Boosting plus targeted train-only augmentation | 58.7% | 58.5% | 47.1% | 46.7% |
| BYOL-A audio embeddings + MLP ANN | 60.0% | 59.3% | 53.0% | 52.6% |
| Log-mel CNN | — | — | 48.5% | 47.0% |

The boosting model's validation accuracy (58.9%) is the pre-augmentation score; after augmentation, validation accuracy was 58.7%. The 50-trial best configuration was selected by validation macro-F1. No test metric was used for tuning or model selection. For the direct comparison, BYOL-A and SVM both score on the same 785 test tracks: BYOL-A is 53.0% accurate versus 49.3% for SVM.

## Optuna configuration

Optuna ran 50 trials, maximizing validation macro-F1. Best trial: 36.

```json
{
  "max_iter": 175,
  "learning_rate": 0.0924952103,
  "max_leaf_nodes": 24,
  "min_samples_leaf": 25,
  "l2_regularization": 2.5736848023,
  "max_bins": 160
}
```

Feature engineering retained the original 518 columns and added 33 family-level row summaries (mean, standard deviation, and range), for 551 features total.

## Validation error analysis and augmentation

The weakest validation recalls were Pop (20%) and Instrumental (40%); those two training classes were augmented with jittered replicas at 2.5% of each feature's robust scale. Augmentation raised validation macro-F1 from 58.503% to 58.534%—a 0.03 percentage-point change—while validation accuracy fell from 58.94% to 58.69%. Treat that tiny macro-F1 change as negligible rather than evidence of a reliable gain.

The selected augmented model reached 47.1% accuracy / 46.7% macro-F1 on the held-out test set. The validation-to-test drop shows that the boosted model did not generalize well enough to replace the BYOL-A ANN. The augmentation did not achieve the requested 80% accuracy, and the overall results do not support promising that target on this dataset with these features.

## Why accuracy is stuck

1. The 518 published descriptors summarize whole tracks and discard detailed temporal structure.
2. Several genres have overlapping sound; the weakest classes are particularly ambiguous.
3. FMA Small's official track-level test split is a separate generalization check. The boosted model scored well on validation but dropped on test, indicating overfitting to that validation split/configuration.
4. The feature rows are nearly balanced and complete, so rebalancing or imputing missing values is unlikely to explain the gap.

## Recommendation

Use BYOL-A embeddings with an MLP ANN as the project's best measured approach so far. It is the strongest tested model on the FMA Small test set. Present the feature-engineered Gradient Boosting experiment as a useful comparison and explain its test-set drop honestly. The published study reporting 64.3% on FMA Small with BYOL-A is a research reference, not a result reproduced by this project. Do not claim 80% or 90% accuracy unless a future run measures it on the same official split.

## Artifacts

- `baseline_metrics.json`: baseline validation score and recorded limiting factors.
- `dataset_analysis.json`: dataset size, split counts, class counts, feature counts, and missingness.
- `optuna_trials.csv` and `optuna_best.json`: all 50 trials and selected parameters.
- `engineered_feature_manifest.json`: feature additions.
- `validation_error_analysis.json`, `validation_class_errors.csv`, and `validation_misclassifications.csv`: validation error study and augmentation comparison.
- `final_test_metrics.json` and `final_test_confusion_matrix.csv`: one held-out test evaluation.
- `feature_engineered_hgb.joblib`: fitted model artifact (do not commit this generated binary to GitHub).
