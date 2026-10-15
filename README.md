# Music Genre Classification

Mini project comparing audio-based machine-learning approaches for classifying FMA Small tracks into eight genres. The project changes the reference paper's CNN/MobileNetV2 approach by comparing published audio descriptors with a pretrained BYOL-A audio encoder and neural-network classifier.

## Dataset

Download and extract FMA Small and FMA metadata separately. The training scripts expect:

- Audio: `Downloads/fma_small/fma_small`
- Metadata: `Downloads/fma_metadata/fma_metadata`

The audio and metadata are not stored in this repository. FMA Small contains 8,000 30-second tracks across eight balanced genres. The scripts preserve FMA's official training, validation, and test splits.

## Measured results

On the official FMA Small split, the BYOL-A pretrained audio embeddings + MLP model achieved 53.0% test accuracy. An RBF SVM on 518 published descriptors achieved 49.3% test accuracy on the same 785 test tracks. The feature-engineered Gradient Boosting experiment reached 58.9% validation accuracy after 50 Optuna trials, but 47.1% test accuracy after the validation-selected augmentation. These runs did not achieve 80% or 90% accuracy.

Detailed results and limitations: [`training_report.md`](outputs/music_genre_training/advanced_results/training_report.md).

## Train the feature-based models

Install Python dependencies from `outputs/music_genre_training/audio_requirements.txt`, then run from the project root:

```powershell
python outputs/music_genre_training/train_fma_experiments.py --trials 50
```

To train the pretrained BYOL-A audio model, see [`outputs/music_genre_training/README.md`](outputs/music_genre_training/README.md). Its encoder files must be obtained separately; review the upstream license before use or redistribution.

## Repository contents

- `outputs/music_genre_training/`: source scripts, requirements, reports, and compact metrics/plots.
- `outputs/music_genre_training/advanced_results/`: baseline diagnostics, Optuna trial table, validation error analysis, and final test metrics.
- `.gitignore`: excludes audio data, model checkpoints, embeddings, caches, and local environments.
