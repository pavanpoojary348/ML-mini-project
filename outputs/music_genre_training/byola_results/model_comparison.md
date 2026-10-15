# Music genre model comparison

## Result on the same songs

Both models were evaluated on the same **785 FMA Small test tracks**, using the dataset's official test split.

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| FMA precomputed features + RBF-SVM | 49.3% | 0.490 |
| BYOL-A pretrained audio embeddings + MLP | 53.0% | 0.526 |

BYOL-A improves accuracy by **3.7 percentage points** on this common set. Its full test-set score is 53.0% accuracy and 0.526 macro-F1 across 800 tracks.

## Why this is below the reference paper's 98%

The reference paper describes GTZAN and an 80/20 train/test split. This project uses FMA Small and its separate official train, validation, and test splits. The paper does not provide enough detail to reproduce its split or rule out overlap between related audio samples, and its text reports several different scores (90.27%, 98%, 98.43%, and 94.20%). These figures should not be compared as though they came from the same dataset and evaluation procedure.

We selected BYOL-A based on research reporting stronger FMA-Small results for pretrained audio embeddings with a neural classifier. This implementation uses a simple MLP and six crops per song; it does not reproduce the paper's full classifier and auxiliary-loss setup, so its published score is a reference, not our result.

## Training details

- Data: 7994 decodable tracks from FMA Small (six audio files could not be decoded).
- Genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock.
- Splits: 6394 train, 800 validation, 800 test.
- Six evenly spaced 0.95-second log-mel crops per song; normalization statistics come from training audio only.
- BYOL-A pretrained encoder is frozen. The MLP epoch is selected on validation macro-F1, then refit on train plus validation before the single test evaluation.

## References

- [Music Genre Classification Using CNN and MobileNetV2](C:/Users/pooja/Downloads/Music_Genre_Classification_Using_CNN_and_MobileNetV2.pdf)
- [BYOL-A music genre study](https://arxiv.org/abs/2603.13871)
- [Official BYOL-A implementation](https://github.com/nttcslab/byol-a)
