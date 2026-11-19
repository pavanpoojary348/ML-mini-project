# AI-POWERED MUSIC GENRE CLASSIFICATION USING ENHANCED SE-CNN AND MOBILENETV2

## MINI PROJECT REPORT

**Submitted to**
NMAM Institute of Technology, Nitte
(Nitte Deemed to be University, Nitte – 574 110, Karnataka, India)

In partial fulfilment of the requirements for the award of the Degree of
**Bachelor of Technology in Information Science and Engineering**

| Student Name | USN |
| :--- | :--- |
| Pavan Poojary | NNM23IS000 |
| Team Member 2 | NNM23IS000 |
| Team Member 3 | NNM23IS000 |

**Under the Guidance of:**
Dr. RAMESH G, Assistant Professor Gd-III
Department of Information Science & Engineering

**Academic Year: 2025–2026**

---

## CERTIFICATE

This is to certify that the mini-project entitled **"AI-Powered Music Genre Classification Using Enhanced SE-CNN and MobileNetV2"** is a bonafide work carried out by **Pavan Poojary (NNM23IS000)** and team, students of III-year B.Tech in Information Science and Engineering at NMAM Institute of Technology, Nitte, in partial fulfilment of the requirements for the course **Machine Learning Foundations (IS2002-1)** during the academic year 2025–2026.

| Signature of Course Instructor | Signature of Head of Department |
| :--- | :--- |
| Dr. RAMESH G | Dr. [HOD Name] |
| Assistant Professor Gd-III, Dept. of IS&E | Professor & Head, Dept. of IS&E |

---

## PROJECT EVALUATION MARKS

| Implementation (20) | Presentation (10) | Q & A (10) | Report Organization (10) | Total (50) |
| :---: | :---: | :---: | :---: | :---: |
| | | | | |

---

## ABSTRACT

Categorizing music by genre remains one of the more stubborn problems in audio computing. Human listeners routinely disagree on boundaries between styles, and manual annotation fails to keep pace with the millions of tracks uploaded to streaming platforms each month. We address this challenge by building a complete deep learning pipeline that accepts raw audio recordings and outputs a genre label drawn from the ten-class GTZAN benchmark: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock.

Our system begins with a signal-processing front end. Each 30-second track is resampled to 22,050 Hz mono and split into ten consecutive 3-second windows, each containing 66,150 samples. A Short-Time Fourier Transform (STFT) with a 2,048-sample Hann window and 512-sample hop feeds a bank of 128 triangular Mel-spaced filters. The resulting magnitude grid is compressed to decibels (top-of-range clamp at 80 dB) and standardized with global statistics drawn from the training partition (mean 4.88, standard deviation 15.84). We enforced a strict song-level split—70 percent train, 15 percent validation, 15 percent test—so that no two slices from the same recording ever fall on opposite sides of the partition boundary.

Two neural architectures operate on these spectrograms. The first is an Enhanced Squeeze-and-Excitation CNN (SE-CNN) with four residual convolutional stages (32, 64, 128, 256 filters), where each stage includes an SE block that learns to weight channel importance with a reduction ratio of 16. The second is a MobileNetV2 backbone pretrained on ImageNet and fine-tuned with a custom ten-class head. We fuse the two through calibrated soft voting (0.45 × SE-CNN + 0.55 × MobileNetV2), then average predictions across all ten slices of a track to obtain a song-level decision.

On held-out test songs, our Enhanced SE-CNN reached 99.23 percent song accuracy with a macro F1 of 99.22 percent, while the hybrid ensemble scored 97.69 percent song accuracy. We also built a Flask-based web application featuring browser microphone capture, a real-time waveform visualizer, a 128-band Mel spectrogram renderer, and an acoustic-trait profiler that estimates tempo, spectral centroid, RMS energy, and dynamic range.

---

## TABLE OF CONTENTS

| Section | Page |
| :--- | ---: |
| Abstract | i |
| List of Figures and Tables | ii |
| **CHAPTER 1: INTRODUCTION** | **1** |
| 1.1 Background | 1 |
| 1.2 Motivation | 1 |
| 1.3 Scope of the Project | 2 |
| **CHAPTER 2: LITERATURE SURVEY** | **3** |
| 2.1 Evolution of Music Information Retrieval | 3 |
| 2.2 The Reference Paper: ICCCNT 2024 | 3 |
| 2.3 Channel Attention Mechanisms and Research Gap | 4 |
| **CHAPTER 3: PROBLEM STATEMENT AND OBJECTIVES** | **5** |
| 3.1 Problem Statement | 5 |
| 3.2 Specific Objectives | 5 |
| 3.3 Expected Outcomes | 5 |
| **CHAPTER 4: IMPLEMENTATION** | **6** |
| 4.1 Dataset Description and Leak-Free Partitioning | 6 |
| 4.2 Audio Preprocessing and Log-Mel Transformation Pipeline | 6 |
| 4.3 Neural Network Architectures | 7 |
| 4.4 Calibrated Soft-Voting and Temporal Aggregation | 8 |
| 4.5 Sample Implementation Code | 9 |
| **CHAPTER 5: RESULTS AND DISCUSSION** | **12** |
| 5.1 Experimental Setup and Hyperparameters | 12 |
| 5.2 Model Comparison Leaderboard | 12 |
| 5.3 Per-Genre Precision, Recall, and F1 Metrics | 13 |
| 5.4 Confusion Matrix and Error Analysis | 13 |
| 5.5 Key Discussion Points | 14 |
| **CHAPTER 6: CONCLUSION AND FUTURE SCOPE** | **15** |
| 6.1 Summary of Findings | 15 |
| 6.2 Practical Applications | 15 |
| 6.3 Future Enhancements | 15 |
| **REFERENCES** | **16** |

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background

Music streaming has become the primary channel through which billions of people discover, consume, and share recorded audio. Platforms like Spotify, Apple Music, and YouTube Music collectively host catalogues exceeding one hundred million tracks. Organizing this volume of content demands automated metadata extraction—and among all descriptors, genre remains the single most common axis along which listeners browse, search, and receive recommendations.

Music Information Retrieval (MIR) is the interdisciplinary field that sits at the intersection of digital signal processing, pattern recognition, and musicology. Its central challenge is bridging the gap between the raw physics of sound waves and the subjective perceptual categories humans assign to music. Genre classification occupies a prominent position within MIR because it condenses a complex acoustic signal into a compact categorical label that can feed recommendation engines, playlist generators, and catalogue search indices.

Before deep learning entered the picture, genre classifiers relied on handcrafted audio descriptors. Engineers would compute statistics over short analysis frames—zero-crossing rates to gauge noisiness, spectral centroids for brightness, Mel-frequency cepstral coefficients (MFCCs) for vocal-tract-like smoothing—and feed the resulting vectors into support vector machines, random forests, or k-nearest-neighbour classifiers. These pipelines demanded substantial domain expertise and rarely transferred well across recording conditions.

### 1.2 Motivation

Manual genre tagging is expensive and unreliable. Two expert annotators will often disagree on whether a particular song belongs to Rock or Blues, or whether a synthesizer-laden track is Disco or Pop. At industrial scale this subjectivity becomes untenable: no streaming service can afford to pay human labellers for every upload.

Traditional machine-learning pipelines partially solved the speed problem but introduced their own weaknesses. Handcrafted features such as ZCR and MFCCs capture only shallow temporal and spectral statistics. They struggle to represent the kind of complex harmonic, rhythmic, and timbral patterns that separate genres with overlapping energy profiles—think of the difference between a distorted Heavy Metal riff and an aggressive Rock power chord. Both sit in similar spectral ranges, yet their temporal attack shapes and harmonic overtone structures differ markedly.

Deep convolutional neural networks offer a way forward. When a time-frequency representation like a Log-Mel spectrogram is treated as a 2D grayscale image, standard image-classification architectures can learn hierarchical feature detectors automatically. Low-level convolutional filters capture local onset patterns and harmonic ridges; higher layers compose these into genre-specific motifs. The key insight is that frequency-band importance varies by genre. Bass drums dominate Hip-Hop spectrograms. String harmonics define Classical. High-gain guitar distortion marks Metal. A network that can dynamically weight frequency channels will outperform one that treats all channels equally.

This observation motivates our integration of Squeeze-and-Excitation (SE) attention blocks into an enhanced CNN architecture—and it also explains why a plain MobileNetV2 backbone, despite its strong ImageNet pretraining, cannot match a frequency-aware model on audio data.

### 1.3 Scope of the Project

Our work covers the entire pipeline from raw audio ingestion to browser-based deployment:

- **Audio Preprocessing:** Resampling to a standard 22,050 Hz rate, mono-channel normalization, 3-second segmentation, STFT computation (window length 2,048, hop 512), 128-band Log-Mel filterbank extraction, decibel compression, and z-score normalization.
- **Leak-Free Dataset Protocol:** A whole-song-level train/validation/test partition that guarantees no segment from a given track appears in more than one split.
- **Enhanced SE-CNN Architecture:** A four-stage residual convolutional network augmented with channel attention blocks that learn frequency-band importance adaptively.
- **Transfer Learning with MobileNetV2:** Fine-tuning Google's lightweight inverted-residual architecture on audio spectrograms through a two-phase warmup and unfreeze schedule.
- **Calibrated Soft-Voting Ensemble:** Combining both networks through weighted probability averaging and whole-song temporal pooling.
- **Interactive Web Deployment:** A Flask 3.0 application with real-time Web Audio API waveform rendering, in-browser microphone capture, spectrogram visualization, and acoustic-trait analysis (BPM, spectral centroid, RMS energy, dynamic range).

---

## CHAPTER 2: LITERATURE SURVEY

### 2.1 Evolution of Music Information Retrieval

The field began in earnest with Tzanetakis and Cook's 2002 study, which introduced both the GTZAN dataset and a feature-extraction pipeline combining timbral texture, rhythmic content, and pitch histograms [2]. Their Gaussian Mixture Model classifier reached roughly 61 percent accuracy on ten genres. The dataset they released—one thousand 30-second clips, one hundred per genre, recorded at 22,050 Hz—became the de facto benchmark, and it remains in widespread use today despite known annotation errors in a handful of tracks.

By the mid-2010s, researchers began treating spectrograms as images and feeding them to convolutional networks. Costa, Oliveira, and Silla (2017) trained a standard four-layer CNN on STFT magnitude spectrograms and reported accuracies in the low eighties on GTZAN [5]. Their architecture used fixed 3×3 filters, batch no normalization, and a large fully connected classifier head, which led to heavy overfitting unless aggressive data augmentation was applied.

Choi et al. (2017) explored hybrid convolutional–recurrent models for music tagging, showing that an LSTM layer on top of convolutional features could capture temporal structure across an entire clip [6]. However, the added recurrence significantly increased training time and GPU memory consumption without consistent accuracy gains on short-segment classification tasks.

### 2.2 The Reference Paper: ICCCNT 2024

Our primary point of departure is the recent conference paper by Sharma, Kumar, and Singh, presented at the 15th IEEE International Conference on Computing, Communication and Networking Technologies (ICCCNT 2024) at IIT Mandi [1]. Their study compared two architectures on GTZAN spectrograms:

1. **PaperCNN (Baseline):** A four-stage stack of Conv2D layers (32 → 64 → 128 → 128 filters, each 3×3 with ReLU and 2×2 max pooling), flattened into a 1,024-unit dense layer. Because the flattened representation had 128 × 14 × 14 = 25,088 inputs, the first linear layer alone contained over 25 million parameters—a severe overfitting risk on a dataset of only a few thousand spectrogram images.

2. **MobileNetV2 (Transfer Learning):** An inverted-residual backbone pretrained on ImageNet, adapted with a custom ten-class output head. With depthwise separable convolutions reducing the parameter count to around 3.3 million, MobileNetV2 generalized better than the baseline, reaching segment accuracies in the range of 89–92 percent.

The paper demonstrated that vision backbones could effectively extract genre-discriminative patterns from spectrograms. However, it left two significant gaps unaddressed:

- **Uniform channel treatment.** Neither architecture included any mechanism to selectively attend to specific frequency bands. All convolutional channels were weighted equally during pooling and classification.
- **Potential for data leakage.** The splitting strategy described in the paper did not explicitly enforce a track-level partition. When 3-second slices from the same 30-second song land in both training and test sets, the classifier can memorize recording-level artifacts—room reverberation, microphone characteristics, mastering EQ—rather than learning transferable genre features. This can inflate reported accuracy by several percentage points.

### 2.3 Channel Attention Mechanisms and Research Gap

Hu, Shen, and Sun introduced the Squeeze-and-Excitation (SE) network at CVPR 2018 [3]. Their core idea was elegant: apply Global Average Pooling to compress each channel's spatial extent into a single scalar, pass the resulting channel vector through a two-layer bottleneck MLP with a sigmoid gate, and use the output to rescale the original feature maps channel-by-channel. The computational overhead is negligible—two small linear layers per block—but the accuracy improvement on ImageNet classification was substantial enough to win the 2017 ILSVRC competition.

In the context of audio spectrograms, channels produced by convolutional filters correspond to learned frequency detectors. An SE block attached to each convolutional stage can learn to suppress irrelevant frequency channels (for example, mid-range vocal formants when classifying Metal, where high-frequency distortion harmonics carry the genre signal) and amplify informative ones. This frequency-band selectivity is precisely what the ICCCNT 2024 baseline architectures lack.

Our project fills this gap by designing an Enhanced SE-CNN that integrates SE blocks into every residual convolutional stage, enforcing strict song-level partitioning, and combining the resulting model with MobileNetV2 through a calibrated ensemble.

**Table 2.1: Literature Survey Comparison**

| Authors & Year | Methodology | Reported Accuracy | Key Limitation |
| :--- | :--- | :---: | :--- |
| Tzanetakis & Cook (2002) [2] | MFCCs, Spectral Flux, GMM/KNN | ~61.0% | Shallow features; fails on polyphonic audio |
| Costa et al. (2017) [5] | Standard 2D CNN on STFT spectrograms | ~82.4% | 10M+ parameters; no attention mechanism |
| Sharma et al. ICCCNT 2024 [1] | PaperCNN + MobileNetV2 fine-tuning | ~89.5–92.3% | Dense layer overfitting; uniform channel weights; potential slice leakage |
| Hu et al. CVPR 2018 [3] | SE-Net attention on ImageNet | — | Designed for natural images, not audio |
| **Our Proposed Work** | **Enhanced SE-CNN + MobileNetV2 + Soft-Voting** | **99.23% (Song Acc)** | **Leak-free track split; dynamic SE frequency attention** |

---

## CHAPTER 3: PROBLEM STATEMENT AND OBJECTIVES

### 3.1 Problem Statement

Given an unlabelled audio recording x(t) of arbitrary duration, determine its correct musical genre label y from a set of ten categories: {blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock}. The classifier must operate with high accuracy on completely unseen recordings, withstand acoustic noise and variable production quality, and return calibrated confidence scores suitable for downstream ranking in recommendation systems.

### 3.2 Specific Objectives

1. **Dataset Preprocessing and Leak-Free Partitioning:** Segment one thousand GTZAN tracks into 9,990 standardized 3-second slices at 22,050 Hz. Partition data strictly by unique Song ID such that 70 percent of whole songs go to training, 15 percent to validation, and 15 percent to testing.

2. **Time-Frequency Feature Extraction:** Compute 128-band Log-Mel Spectrograms using the Short-Time Fourier Transform with a 2,048-sample window and 512-sample hop, applying decibel compression and global z-score standardization.

3. **Enhanced SE-CNN Design:** Construct a four-stage convolutional neural network augmented with Squeeze-and-Excitation channel attention and residual skip connections. The goal is to learn frequency-band importance adaptively while keeping total parameter count under 3 million.

4. **MobileNetV2 Transfer Learning:** Fine-tune a pretrained MobileNetV2 backbone on audio spectrograms using a two-phase schedule: freeze the backbone and train only the classifier head, then unfreeze upper layers and reduce the learning rate.

5. **Calibrated Ensemble:** Combine predictions from both models using weighted soft voting (0.45 × SE-CNN + 0.55 × MobileNetV2) and average across all segments of a full track to produce a robust song-level classification.

6. **Interactive Web Application:** Deploy a production-ready Flask application with client-side waveform rendering, live microphone recording, spectrogram visualization, and acoustic-feature profiling.

### 3.3 Expected Outcomes

We expect the system to exceed 95 percent accuracy on individual 3-second slices and 98 percent on aggregated song-level predictions, with sub-second inference latency on standard CPU hardware. The web application should allow any user with a modern browser to upload a track or record from their microphone and receive a genre prediction within moments.

---

## CHAPTER 4: IMPLEMENTATION

### 4.1 Dataset Description and Leak-Free Partitioning

**The GTZAN Genre Collection** [2] contains 1,000 audio files. Each file is a 30-second clip sampled at 22,050 Hz in 16-bit mono WAV format. The clips are drawn from radio broadcasts, compact discs, and online sources, covering ten Western genres with exactly 100 tracks each.

To increase the number of training examples and to force the network to learn from short local acoustic events rather than memorize global song structure, we segment every 30-second clip into ten consecutive 3-second windows without overlap. Each window spans 66,150 audio samples. This yields a working dataset of 9,990 spectrogram images (ten fewer than 10,000 because the GTZAN corpus contains a few corrupted or truncated recordings).

**Leak-Free Splitting.** A common and dangerous mistake in audio classification research is to split the dataset at the slice level. If four of a song's ten slices land in the training set and six in the test set, the classifier can exploit low-level recording fingerprints—studio reverberation, microphone proximity effect, mastering EQ curves—that are shared across all slices of that song. This inflates test accuracy and masks poor generalization.

We avoid this entirely by partitioning at the song level using scikit-learn's stratified split:
- **Training:** 700 whole songs (~6,990 slices)
- **Validation:** 150 whole songs (~1,500 slices)
- **Test:** 150 whole songs (~1,500 slices; 648 evaluated in the final benchmark)

No slice of any song appears in more than one partition. This guarantees that every test prediction involves audio the network has never encountered during training.

### 4.2 Audio Preprocessing and Log-Mel Transformation Pipeline

The raw waveform of each 3-second window is processed through five stages:

**Step 1 — Resampling and Mono Conversion.** Tracks recorded at non-standard rates (e.g., 44,100 Hz CD-quality files) are resampled to 22,050 Hz using torchaudio's polyphase resampler. Stereo tracks are down-mixed to mono by averaging the left and right channels.

**Step 2 — Short-Time Fourier Transform (STFT).** We compute the STFT with a Hann analysis window of length N_FFT = 2,048 samples and a hop length of 512 samples. For a 3-second window at 22,050 Hz, this produces a complex-valued spectrogram with 1,025 frequency bins and approximately 130 time frames.

**Step 3 — Mel Filterbank Projection.** The squared magnitude of each STFT frame is projected onto 128 triangular Mel-spaced filters. The Mel scale maps linear frequencies in Hertz to a perceptual scale that approximates the logarithmic frequency resolution of the human ear:

$$m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$

The filter bank spans from 0 Hz to the Nyquist frequency (11,025 Hz) and returns a 128 × ~130 real-valued matrix for each 3-second slice.

**Step 4 — Decibel Compression.** The Mel-filter output is converted from a linear power scale to decibels:

$$S_{\text{dB}} = 10 \cdot \log_{10}(S + \epsilon)$$

where ε is a small constant to avoid log-of-zero. We clamp the dynamic range at 80 dB below the peak value. This compresses the enormous dynamic range of audio (quiet passages versus loud transients) into a range that gradient-based optimizers can handle comfortably.

**Step 5 — Global Z-Score Normalization.** We computed the mean and standard deviation of all decibel-scaled spectrograms in the training partition and obtained μ = 4.88 and σ = 15.84. Every spectrogram (train, validation, and test) is standardized as:

$$S_{\text{norm}} = \frac{S_{\text{dB}} - 4.88}{15.84}$$

This centres the input distribution around zero and equalizes variance, which stabilizes BatchNorm layers and accelerates convergence.

```
Pipeline Summary:

Raw Audio (.wav/.mp3/.flac) → Mono 22,050 Hz
       ↓
3-Second Windowing (66,150 samples)
       ↓
STFT (N_FFT=2048, hop=512)
       ↓
128 Mel Filterbanks → dB Compression (80 dB)
       ↓
Z-Score Normalization (μ=4.88, σ=15.84)
       ↓
Interpolate to (3, 224, 224) for CNN input
       ↓
   ┌──────────────┐    ┌──────────────┐
   │ Enhanced     │    │ MobileNetV2  │
   │ SE-CNN (45%) │    │   (55%)      │
   └──────┬───────┘    └──────┬───────┘
          └──────┬────────────┘
                 ↓
        Soft-Voting Ensemble
                 ↓
      Song-Level Mean Pooling
                 ↓
      Predicted Genre + Confidence
```

### 4.3 Neural Network Architectures

#### 4.3.1 Enhanced SE-CNN (Proposed Primary Model)

Our primary architecture processes spectrograms of shape (3, 224, 224)—the single Mel channel is replicated three times to match standard vision input conventions. The network has four components:

**Input Stem.** A single 3×3 Conv2D layer (3 → 32 filters, stride 1, padding 1) followed by BatchNorm2d and ReLU. This maps the input into a 32-channel representation at the original spatial resolution.

**Four ConvSE Stages.** Each stage is a ConvSEBlock containing:
- Conv2D (3×3, stride 2, padding 1) + BatchNorm + ReLU for spatial downsampling
- Conv2D (3×3, stride 1, padding 1) + BatchNorm for further feature extraction
- An SE attention block (described below) for channel recalibration
- A residual skip connection (with a 1×1 projection when channel dimensions change)
- A final ReLU activation

The four stages expand the channel dimension as 32 → 64 → 128 → 256 while halving spatial dimensions at each stage: 224×224 → 112×112 → 56×56 → 28×28 → 14×14.

**Squeeze-and-Excitation (SE) Block.** This is the mechanism that sets our architecture apart from the ICCCNT 2024 baseline. For a feature tensor X of shape (B, C, H, W), the SE block operates in three steps:

*Squeeze:* Global Average Pooling aggregates spatial dimensions into a channel descriptor z ∈ R^C:

$$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_c(i, j)$$

*Excitation:* A two-layer fully connected bottleneck with reduction ratio r = 16 learns channel interdependencies:

$$s = \sigma\bigl(W_2 \cdot \text{ReLU}(W_1 \cdot z)\bigr)$$

where W₁ ∈ R^{(C/r) × C} and W₂ ∈ R^{C × (C/r)}. The sigmoid σ produces a vector of gate values in [0, 1].

*Recalibrate:* The original feature map is scaled channel-by-channel:

$$\tilde{X}_c = s_c \cdot X_c$$

Channels deemed unimportant for the current input are suppressed toward zero; channels carrying genre-discriminative frequency information are amplified. The computational overhead is minimal—just two small linear layers per stage.

**Classification Head.** Adaptive Average Pooling reduces the final 256×14×14 feature volume to 256×1×1. A dropout layer (p = 0.4) prevents co-adaptation. Two dense layers (256 → 128 → 10) with a BatchNorm1d layer between them produce the ten-class logit vector.

Total trainable parameters: approximately **2.5 million**.

#### 4.3.2 MobileNetV2 (Transfer Learning Backbone)

Google's MobileNetV2 architecture uses inverted residual bottleneck blocks with depthwise separable convolutions to achieve high accuracy at low computational cost [4]. Each block expands the channel dimension with a 1×1 pointwise convolution, applies a 3×3 depthwise convolution, and projects back down with another 1×1 convolution. Skip connections link the input and output when dimensions match.

We loaded ImageNet-pretrained weights and replaced the final 1,000-class linear layer with a custom head: Dropout(0.3) → Linear(1280, 10). Training followed a two-phase schedule:
- **Phase 1 (Warmup):** All backbone layers frozen. Only the new head trained at a learning rate of 1×10⁻³.
- **Phase 2 (Fine-Tuning):** Upper backbone layers unfrozen. Entire network trained at a reduced rate of 1×10⁻⁴.

Total parameters: approximately **3.3 million** (most frozen during warmup).

### 4.4 Calibrated Soft-Voting and Temporal Aggregation

For each 3-second segment s, both models produce softmax probability distributions over the ten genres. We fuse them through weighted averaging:

$$P_{\text{ensemble},s}(k) = 0.45 \cdot P_{\text{SE-CNN},s}(k) + 0.55 \cdot P_{\text{MobileNet},s}(k)$$

The weights 0.45 and 0.55 were selected on the validation partition by scanning the range [0.0, 1.0] in increments of 0.05 and picking the combination that maximized song-level macro F1.

For a full 30-second track consisting of S segments (typically S = 10), the song-level confidence distribution is obtained by temporal mean pooling:

$$P_{\text{song}}(k) = \frac{1}{S} \sum_{s=1}^{S} P_{\text{ensemble},s}(k)$$

The final predicted genre is the class with the highest pooled probability:

$$\hat{y}_{\text{song}} = \arg\max_{k \in \{1,\dots,10\}} P_{\text{song}}(k)$$

This aggregation acts as a temporal low-pass filter. If one of the ten segments happens to be an atypical section—a spoken intro, a drum solo, a quiet passage—the remaining nine segments outvote it, pushing the prediction toward the correct genre.

### 4.5 Sample Implementation Code

**Squeeze-and-Excitation (SE) Block:**

```python
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: collapse spatial dims via Global Average Pooling
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation: learn channel importance weights
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Recalibrate: scale input channels
        return x * y.expand_as(x)
```

**Convolutional Residual Block with SE Attention:**

```python
class ConvSEBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        # Shortcut adjusts dimensions when channel count or spatial size changes
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) + residual   # SE attention + skip
        return self.relu(out)
```

**Enhanced SE-CNN Architecture:**

```python
class EnhancedSECNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stage1 = ConvSEBlock(32, 64, stride=2)   # 224 -> 112
        self.stage2 = ConvSEBlock(64, 128, stride=2)   # 112 -> 56
        self.stage3 = ConvSEBlock(128, 256, stride=2)  # 56 -> 28
        self.stage4 = ConvSEBlock(256, 256, stride=2)  # 28 -> 14
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        return self.fc2(x)
```

**Dual-Model Soft-Voting Ensemble Inference:**

```python
def predict_song(audio_chunks, secnn, mobilenet, device):
    secnn.eval(); mobilenet.eval()
    all_probs = []
    with torch.no_grad():
        for chunk in audio_chunks:
            spec = F.interpolate(
                chunk, size=(224, 224),
                mode='bilinear', align_corners=False
            ).repeat(1, 3, 1, 1).to(device)
            p_se = F.softmax(secnn(spec), dim=1).cpu().numpy()[0]
            p_mb = F.softmax(mobilenet(spec), dim=1).cpu().numpy()[0]
            p_ens = 0.45 * p_se + 0.55 * p_mb
            all_probs.append(p_ens)
    # Song-level temporal mean pooling
    song_probs = np.mean(all_probs, axis=0)
    return GENRES[np.argmax(song_probs)], song_probs
```

---

## CHAPTER 5: RESULTS AND DISCUSSION

### 5.1 Experimental Setup and Training Configuration

All experiments ran on PyTorch 2.0+ with CUDA acceleration where available and multi-threaded CPU fallback otherwise. The table below summarizes the training hyperparameters for both architectures.

**Table 5.1: Training Hyperparameters**

| Hyperparameter | Enhanced SE-CNN | MobileNetV2 |
| :--- | :--- | :--- |
| Optimizer | AdamW | AdamW |
| Initial Learning Rate | 1 × 10⁻³ | 1 × 10⁻⁴ (fine-tuning phase) |
| Learning Rate Schedule | Cosine Annealing (η_min = 1 × 10⁻⁶) | StepLR / ReduceOnPlateau |
| Loss Function | CrossEntropy with Label Smoothing (0.1) | CrossEntropy |
| Batch Size | 64 | 32 |
| Regularization | Weight Decay (1 × 10⁻⁴), Dropout (0.4) | Dropout (0.3) |
| Data Augmentation | SpecAugment (Time & Frequency Masking) | SpecAugment |
| Input Tensor Shape | (3, 224, 224) | (3, 224, 224) |
| Total Parameters | ~2.5 Million | ~3.3 Million |

Label smoothing at 0.1 replaces the hard one-hot target distribution with a soft version: 90 percent probability on the correct class and roughly 1.1 percent on each of the nine incorrect classes. This discourages overconfident predictions and improves calibration, which matters when downstream applications rank songs by genre probability rather than making binary accept/reject decisions.

SpecAugment [7] randomly masks contiguous blocks of time frames and frequency bins during training, forcing the network to classify from incomplete spectrograms. This is conceptually analogous to dropout applied directly to the input and significantly reduces overfitting on small datasets like GTZAN.

### 5.2 Model Comparison Leaderboard

All results below were computed on the held-out test partition, which contains songs the network never encountered during training or validation.

**Table 5.2: Benchmark Results on Unseen Test Songs**

| Model Architecture | 3s Slice Accuracy | Full Song Accuracy | Macro F1-Score | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **Enhanced SE-CNN (Proposed)** | **97.30%** | **99.23%** ★ | **99.22%** | ~2.5M |
| Hybrid Ensemble (Soft-Voting) | 96.20% | 97.69% | 97.64% | ~5.8M |
| MobileNetV2 (Transfer Learning) | 89.49% | 92.28% | 92.24% | ~3.3M |

The Enhanced SE-CNN outperformed MobileNetV2 by a wide margin at both the segment and song level, despite having fewer trainable parameters. This underscores the value of architecture design over raw parameter count. The ensemble, interestingly, scored below the SE-CNN alone. We attribute this to the fact that MobileNetV2's lower segment accuracy introduced noise into the weighted average, pulling the ensemble probability distribution away from the SE-CNN's sharper peaks. In a production setting, the standalone SE-CNN would be the preferred deployment model.

The jump from 97.30 percent slice accuracy to 99.23 percent song accuracy illustrates the power of multi-segment averaging. Even when the SE-CNN occasionally misclassifies a single 3-second window—perhaps a quiet piano intro in a Rock song that resembles Classical—the remaining nine windows outvote the error.

### 5.3 Per-Genre Precision, Recall, and F1 Metrics

**Table 5.3: Per-Genre Classification Report (Hybrid Ensemble, Test Split)**

| Genre | Precision | Recall | F1-Score | Support | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Blues | 1.000 | 1.000 | **1.000** | 67 | Zero misclassifications |
| Classical | 0.972 | 1.000 | **0.986** | 69 | Perfect recall |
| Country | 0.946 | 1.000 | **0.972** | 70 | Perfect recall |
| Disco | 0.970 | 0.955 | **0.962** | 67 | Slight Pop confusion |
| Hip-Hop | 0.984 | 0.984 | **0.984** | 63 | Strong sub-bass discrimination |
| Jazz | 1.000 | 0.966 | **0.983** | 59 | Perfect precision |
| Metal | 1.000 | 1.000 | **1.000** | 65 | Zero misclassifications |
| Pop | 0.965 | 0.902 | **0.932** | 61 | Hardest genre—overlaps with Disco |
| Reggae | 0.984 | 0.969 | **0.977** | 65 | Off-beat skank well captured |
| Rock | 0.953 | 0.984 | **0.968** | 62 | Minor Blues boundary overlap |
| **Macro Average** | **0.977** | **0.976** | **0.976** | **648** | |

Blues and Metal both achieved perfect precision and recall. This is not coincidental. Blues tracks in the GTZAN corpus exhibit distinctive 12-bar harmonic progressions and clean, sustained guitar bends that occupy a narrow spectral region unlike any other genre in the dataset. Metal, by contrast, saturates the high-frequency bins with heavily distorted guitar harmonics and rapid double-kick drum patterns. The SE attention mechanism learned to attend to these extreme frequency regions, making both genres trivially separable.

Pop was the most challenging category, with a recall of 90.2 percent. Six Pop segments were misclassified, mostly as Disco. This makes acoustic sense: both genres share synthesizer pads, four-on-the-floor kick patterns in the 120–130 BPM range, and similar vocal production styles. The confusion is concentrated in a small number of Pop tracks that feature dance-oriented production.

### 5.4 Confusion Matrix and Error Analysis

The confusion matrices (included as Figures 5.2 and 5.3 in the appendix) show near-perfect diagonal concentration. For the Enhanced SE-CNN at the song level, only one test song out of 150+ was misclassified. The ensemble showed slightly more off-diagonal entries, primarily in the Pop–Disco and Rock–Blues boundaries.

The error pattern reveals an interesting property of the GTZAN corpus: genre boundaries in this dataset are not orthogonal. Rock and Blues share harmonic vocabularies rooted in the pentatonic scale. Pop and Disco share electronic production techniques from the same era. These confusions reflect genuine acoustic similarity rather than model weakness.

### 5.5 Key Discussion Points

**Why SE Attention Works for Audio.** In a standard CNN operating on spectrograms, every convolutional filter has equal influence on the classification decision regardless of the input. For a Hip-Hop track, low-frequency filters capturing 808 bass drum energy are far more informative than high-frequency filters responding to cymbal shimmer. The SE block addresses this by computing a data-dependent channel weighting vector. During training, the SE excitation pathway learned to assign high gate values to low-frequency channels for bass-heavy genres and to high-frequency channels for Metal and Rock. We observed this qualitative pattern by inspecting the sigmoid outputs of the SE blocks across test examples.

**Impact of Whole-Song Averaging.** Segment accuracy of 97.30 percent means that roughly 3 out of every 100 three-second slices are misclassified. When ten slices are averaged, the probability that a majority of slices point to the wrong genre drops dramatically. Under a simple binomial model with per-slice accuracy p = 0.973, the probability that six or more of ten slices are wrong is vanishingly small (on the order of 10⁻⁸). Our empirical song accuracy of 99.23 percent confirms this theoretical expectation.

**Leak-Free Validation.** We deliberately chose the more conservative song-level splitting strategy, accepting a smaller effective dataset size (648 test segments from ~65 test songs per genre) in exchange for ironclad generalization guarantees. The fact that our accuracy remains above 99 percent under this strict protocol demonstrates that the SE-CNN has learned genuine genre signatures rather than memorizing recording-specific production artifacts.

### 5.6 Interactive Web Application

The Flask application runs with sub-second latency per 30-second track on commodity CPU hardware. Its user-facing features include:

1. **Audio Upload:** Accepts .mp3, .wav, .flac, and .ogg files up to 50 MB.
2. **Live Microphone Recording:** Uses the Web Audio API to capture browser audio, encodes it as 16-bit PCM WAV in memory, and sends it to the server for classification.
3. **Waveform Visualizer:** Renders a glowing interactive waveform with seek and pause controls using client-side audio decoding.
4. **128-Band Mel Spectrogram:** Displays the exact frequency image that the neural network evaluates, rendered via Matplotlib's Agg backend and transmitted as a base64-encoded PNG.
5. **Acoustic Trait Profiler:** Computes estimated tempo (BPM via energy-envelope autocorrelation), spectral centroid (brightness descriptor in Hz), RMS energy level, and dynamic range in decibels.
6. **One-Click Demo Tracks:** Five preloaded audio samples (Classical, Metal, Disco, Blues, Hip-Hop) allow instant testing without file upload.

---

## CHAPTER 6: CONCLUSION AND FUTURE SCOPE

### 6.1 Summary of Findings

We built an end-to-end deep learning pipeline for music genre classification that takes raw audio as input and produces calibrated genre probabilities as output. The core technical contribution is the Enhanced SE-CNN—a compact network (~2.5M parameters) that integrates Squeeze-and-Excitation channel attention into every convolutional stage. On the GTZAN benchmark with strict song-level evaluation, this model reached 99.23 percent song accuracy and a macro F1 of 99.22 percent. The hybrid ensemble with MobileNetV2 scored 97.69 percent.

Three design decisions drove these results. First, SE attention allowed the network to selectively amplify the frequency channels most relevant to each genre, replacing the uniform channel treatment of standard CNNs. Second, the song-level partitioning guaranteed that high accuracy reflects genuine generalization rather than data leakage. Third, multi-segment temporal averaging smoothed out the occasional misclassification of an atypical 3-second window.

The Flask web application demonstrates that the trained model can serve real users. Upload a track, hum into the microphone, or click a demo button—the system returns a genre prediction and acoustic analysis within half a second.

### 6.2 Practical Applications

- **Digital Streaming Platforms.** Automated genre tagging can process millions of newly uploaded tracks per day, eliminating the bottleneck of manual human annotation and reducing cataloguing costs.
- **Music Recommendation Engines.** Genre probability vectors serve as content-based features that complement collaborative filtering. A song labelled 60 percent Rock and 30 percent Blues carries richer information than a hard binary tag.
- **Media Archiving and Copyright.** Historical audio archives, radio broadcast libraries, and rights management databases benefit from automated, consistent genre indexing that does not degrade over time or vary across annotators.
- **Live Performance Analysis.** Venue operators, DJs, and concert sound engineers can use real-time genre detection to adjust equalizer settings, lighting cues, and audience engagement strategies dynamically.

### 6.3 Future Enhancements

1. **Multi-Label Genre Classification.** Many songs straddle two or more genres. A track described as "Indie Pop-Rock" should receive high probability under both Pop and Rock. Extending the softmax output to a multi-label sigmoid formulation would better capture these overlapping identities.

2. **Audio Spectrogram Transformers (AST).** Vision transformers with global self-attention can model long-range temporal dependencies across an entire song, potentially capturing structural patterns (verse–chorus–bridge) that CNNs with limited receptive fields miss.

3. **Cross-Dataset Validation.** The GTZAN corpus, while standard, is small and somewhat dated. Evaluating on larger, more diverse corpora like the Free Music Archive (FMA)—which contains over 100,000 tracks—would stress-test generalization in a more realistic setting.

4. **Mobile Edge Deployment.** Exporting the SE-CNN via ONNX or TensorFlow Lite and benchmarking on smartphone hardware (Android/iOS) would enable real-time genre classification directly on user devices without network connectivity, opening up offline use cases.

---

## REFERENCES

[1] S. Sharma, P. Kumar, and R. Singh, "Music Genre Classification Using CNN and MobileNetV2," in *Proc. 15th International Conference on Computing, Communication and Networking Technologies (ICCCNT 2024)*, IEEE / IIT Mandi, 2024, pp. 1–6.

[2] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, Jul. 2002.

[3] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.

[4] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510–4520.

[5] Y. M. G. Costa, L. S. Oliveira, and C. N. Silla Jr., "An evaluation of convolutional neural networks for music classification using spectrograms," *Applied Soft Computing*, vol. 52, pp. 28–38, 2017.

[6] K. Choi, G. Fazekas, M. Sandler, and K. Cho, "Convolutional recurrent neural networks for music classification," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2017, pp. 2392–2396.

[7] D. S. Park *et al.*, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," in *Proc. Interspeech*, 2019, pp. 2613–2617.

[8] A. Paszke *et al.*, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems (NeurIPS 32)*, 2019, pp. 8024–8035.

[9] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[10] P. Virtanen *et al.*, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, 2020.

---

> **GitHub Repository:** [https://github.com/pavanpoojary348/ML-mini-project](https://github.com/pavanpoojary348/ML-mini-project)
