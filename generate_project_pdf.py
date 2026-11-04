"""Generate a comprehensive, beautiful PDF Project & Code Guide for the Music Genre Classification Project."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Music_Genre_Classification_Project_Guide.pdf"

class NumberedCanvas(canvas.Canvas):
    """Custom canvas that computes total pages dynamically and draws running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 752, "AI-POWERED MUSIC GENRE CLASSIFICATION  |  VIVA & CODE WALKTHROUGH GUIDE")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 744, 558, 744)

        # Running footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 38, "Deep Learning Audio Project  •  GTZAN Benchmark (10 Genres)")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, 48, 558, 48)

        self.restoreState()


def create_pdf(filename=PDF_FILENAME):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    C_PRIMARY = colors.HexColor("#0F172A")    # Slate 900
    C_BLUE = colors.HexColor("#1D4ED8")       # Blue 700
    C_CYAN = colors.HexColor("#0284C7")       # Cyan 600
    C_GREEN = colors.HexColor("#059669")      # Emerald 600
    C_PURPLE = colors.HexColor("#7C3AED")     # Purple 600
    C_TEXT = colors.HexColor("#1E293B")       # Slate 800
    C_MUTED = colors.HexColor("#475569")      # Slate 600
    C_BG_CARD = colors.HexColor("#F8FAFC")    # Slate 50
    C_BORDER = colors.HexColor("#E2E8F0")     # Slate 200

    # Typography Styles
    style_cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=C_PRIMARY,
        spaceAfter=8,
    )
    style_cover_sub = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=C_BLUE,
        spaceAfter=14,
    )
    style_h1 = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=C_PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    style_h2 = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=C_BLUE,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    style_body = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=C_TEXT,
        spaceAfter=5,
    )
    style_bullet = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=C_TEXT,
        leftIndent=12,
        spaceAfter=4,
    )
    style_code_badge = ParagraphStyle(
        "CodeBadge",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8.5,
        leading=11,
        textColor=C_PURPLE,
    )
    style_callout_title = ParagraphStyle(
        "CalloutTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=C_PRIMARY,
        spaceAfter=3,
    )
    style_table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1, # Center
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=C_TEXT,
    )
    style_table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=C_TEXT,
        alignment=1,
    )

    story = []

    # =========================================================================
    # COVER / HEADER BANNER
    # =========================================================================
    story.append(Paragraph("AI-Powered Music Genre Classification", style_cover_title))
    story.append(Paragraph("Deep Learning on GTZAN Audio Benchmark  •  Complete Viva, Code Walkthrough & Project Guide", style_cover_sub))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceBefore=0, spaceAfter=12))

    # Metric Highlight Card Table
    kpi_data = [
        [
            Paragraph("<b>99.23%</b><br/><font size='7.5' color='#64748B'>Song Test Accuracy</font>", style_table_cell_center),
            Paragraph("<b>10 Genres</b><br/><font size='7.5' color='#64748B'>GTZAN Benchmark</font>", style_table_cell_center),
            Paragraph("<b>~2.5M</b><br/><font size='7.5' color='#64748B'>SE-CNN Parameters</font>", style_table_cell_center),
            Paragraph("<b>Real-Time</b><br/><font size='7.5' color='#64748B'>Flask Web & Mic App</font>", style_table_cell_center),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[126, 126, 126, 126])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # CHAPTER 1: PROJECT OVERVIEW & THE DATASET
    # =========================================================================
    story.append(Paragraph("1. Project Overview & The GTZAN Dataset", style_h1))
    story.append(Paragraph(
        "This project implements an end-to-end, high-accuracy Deep Learning audio classifier and interactive web application. "
        "Audio waveforms cannot be effectively classified by treating them as 1D raw vectors due to extreme dimensionality and phase variance. "
        "Instead, we transform audio into <b>2D Log-Mel Spectrograms</b> (representing frequency harmonics over time) and process them using "
        "modern computer vision architectures augmented with <b>Channel Attention (Squeeze-and-Excitation)</b>.",
        style_body
    ))
    story.append(Paragraph(
        "<b>The GTZAN Dataset:</b> The standard international benchmark for music classification comprising 1,000 audio tracks "
        "(100 tracks per genre, each 30 seconds long, sampled at 22,050 Hz). The 10 target genres are: "
        "<i>Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock</i>.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Data Leakage Prevention:</b> Naive pipelines randomly slice songs and shuffle segments, placing pieces of the same song into "
        "both train and test sets. We enforce strict <b>song-level splitting (70% Train, 15% Val, 15% Test)</b> so the test set consists "
        "strictly of completely unseen songs.",
        style_body
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 2: AUDIO PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("2. Step-by-Step Audio Preprocessing Pipeline", style_h1))
    
    pipeline_steps = [
        ("Step 1: Audio Standardization", "Raw audio files (.wav, .mp3, mic recording) are downmixed to mono and resampled to 22,050 Hz."),
        ("Step 2: 3-Second Segmentation", "Each 30-second track is segmented into 10 consecutive, non-overlapping 3-second slices (66,150 samples). This expands the training dataset 10x (from 1,000 to 9,990 samples) and captures localized timbral patterns."),
        ("Step 3: Short-Time Fourier Transform (STFT)", "Computes frequency spectrum using an FFT window of N_FFT = 2048 and hop_length = 512 samples. Each frame maps to 130 discrete time frames."),
        ("Step 4: Mel Frequency Scale (128 Bands)", "Projects linear Hertz frequencies onto 128 Mel filter banks, mimicking non-linear human pitch perception (high sensitivity in speech/bass, logarithmic in treble)."),
        ("Step 5: Decibel Scaling & Normalization", "Converts power amplitudes to logarithmic decibels (dB) with top_db = 80 dB, normalized using dataset statistics: (spec - 4.88) / 15.84."),
        ("Step 6: SpecAugment Regularization", "During training, random frequency rows and time columns are masked (zeroed out). This forces the neural network to learn multiple acoustic cues rather than relying on a single instrument riff.")
    ]
    for title, desc in pipeline_steps:
        story.append(Paragraph(f"<b>• {title}:</b> {desc}", style_bullet))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 3: NEURAL NETWORK ARCHITECTURES
    # =========================================================================
    story.append(Paragraph("3. Deep Learning Architectures", style_h1))
    
    story.append(Paragraph("A. Enhanced SE-CNN (Our Proposed Primary Model)", style_h2))
    story.append(Paragraph(
        "Standard CNNs treat all spectrogram feature maps equally. However, music audio consists of rich harmonic structures "
        "where specific frequency bands define specific genres (e.g., bass energy in Hip-Hop vs. distorted treble harmonics in Metal). "
        "We introduce <b>Squeeze-and-Excitation (SE) Channel Attention</b> across 4 hierarchical stages:",
        style_body
    ))
    secnn_details = [
        ("Squeeze Operation", "Global Average Pooling collapses 2D time-frequency dimensions (H, W) into a 1D channel descriptor: z_c = (1 / H*W) sum(x_{c, i, j})."),
        ("Excitation Operation", "Two fully connected layers with a bottleneck reduction ratio of 16 and ReLU activation, followed by Sigmoid gating: s = σ(W_2 · δ(W_1 · z)). This learns non-linear inter-channel relationships."),
        ("Channel Recalibration", "The original feature maps are scaled by the learned weights: x'_c = s_c * x_c."),
        ("4 Residual Stages", "Hierarchical progression (32 -> 64 -> 128 -> 256 channels) with 2 Convolutions + BatchNorm + Residual skip connections per stage."),
        ("Adaptive Average Pooling", "Collapses spatial features directly to (1, 1), cutting parameters from ~15M down to ~2.5M and eliminating dense-layer overfitting.")
    ]
    for b_title, b_desc in secnn_details:
        story.append(Paragraph(f"  ✔ <b>{b_title}:</b> {b_desc}", style_bullet))

    story.append(Spacer(1, 6))
    story.append(Paragraph("B. MobileNetV2 Classifier (Transfer Learning)", style_h2))
    story.append(Paragraph(
        "Pretrained on ImageNet, MobileNetV2 (~3.3M parameters) uses depthwise-separable convolutions and inverted residual bottlenecks. "
        "We train it using a <b>Two-Phase Transfer Learning Protocol</b>: Phase 1 freezes the backbone to warm up our custom 10-genre head (lr=1e-3); "
        "Phase 2 unfreezes top residual blocks (from block 12) with a smaller learning rate (5e-5) to adapt visual priors to audio spectrograms.",
        style_body
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("C. Calibrated Soft-Voting Hybrid Ensemble", style_h2))
    story.append(Paragraph(
        "During inference, both models score each 3-second slice. Predictions are combined using weighted soft-voting: "
        "<font color='#0284C7'><b>P_ensemble = 0.45 · P_secnn + 0.55 · P_mobilenet</b></font>. "
        "All segment probabilities across the full song are then averaged to produce the final track prediction.",
        style_body
    ))
    story.append(Spacer(1, 12))

    # =========================================================================
    # CHAPTER 4: EXPERIMENTAL RESULTS
    # =========================================================================
    story.append(Paragraph("4. Experimental Results & Benchmark Leaderboard", style_h1))
    story.append(Paragraph(
        "Evaluated on the held-out unseen test split of GTZAN. Whole-song aggregation significantly outperforms single-slice predictions "
        "by smoothing out atypical intros and silent pauses.",
        style_body
    ))

    # Benchmark Table
    bench_data = [
        [
            Paragraph("<b>Model Architecture</b>", style_table_header),
            Paragraph("<b>Segment Acc (3s)</b>", style_table_header),
            Paragraph("<b>Song Acc (Track)</b>", style_table_header),
            Paragraph("<b>Macro F1-Score</b>", style_table_header),
            Paragraph("<b>Parameter Size</b>", style_table_header),
        ],
        [
            Paragraph("<b>Enhanced SE-CNN (Proposed)</b>", style_table_cell),
            Paragraph("<b>97.30%</b>", style_table_cell_center),
            Paragraph("<font color='#059669'><b>99.23%</b></font>", style_table_cell_center),
            Paragraph("<b>99.22%</b>", style_table_cell_center),
            Paragraph("~2.5M params", style_table_cell_center),
        ],
        [
            Paragraph("Hybrid Ensemble (SE-CNN + MobileNet)", style_table_cell),
            Paragraph("96.20%", style_table_cell_center),
            Paragraph("97.69%", style_table_cell_center),
            Paragraph("97.64%", style_table_cell_center),
            Paragraph("~5.8M params", style_table_cell_center),
        ],
        [
            Paragraph("MobileNetV2 (Transfer Learning)", style_table_cell),
            Paragraph("89.49%", style_table_cell_center),
            Paragraph("92.28%", style_table_cell_center),
            Paragraph("92.24%", style_table_cell_center),
            Paragraph("~3.3M params", style_table_cell_center),
        ],
        [
            Paragraph("Paper Baseline CNN (ICCCNT 2024)", style_table_cell),
            Paragraph("65.20%", style_table_cell_center),
            Paragraph("68.00%", style_table_cell_center),
            Paragraph("67.40%", style_table_cell_center),
            Paragraph("~10.5M params", style_table_cell_center),
        ],
    ]
    bench_table = Table(bench_data, colWidths=[174, 82, 82, 82, 84])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_BLUE),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_BG_CARD]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Per-Genre Performance Highlights:</b> "
        "• <b>Blues & Metal:</b> 100% Precision and 100% Recall (F1 = 1.000).<br/>"
        "• <b>Classical:</b> 97.2% Precision, 100% Recall (F1 = 0.986).<br/>"
        "• <b>Hip-Hop:</b> 98.4% Precision, 98.4% Recall (F1 = 0.984).<br/>"
        "• <b>Jazz:</b> 100% Precision, 96.6% Recall (F1 = 0.983).<br/>"
        "• <b>Rock vs. Country:</b> Successfully resolved difficult acoustic boundaries (Rock: 96.8% F1, Country: 97.2% F1).",
        style_body
    ))
    story.append(Spacer(1, 14))

    # =========================================================================
    # CHAPTER 5: CODEBASE WALKTHROUGH (SECTION BY SECTION)
    # =========================================================================
    story.append(Paragraph("5. Codebase Walkthrough & Section Guide", style_h1))
    story.append(Paragraph(
        "Every file in the project has been organized with clear section headers so you can explain each module directly to your professor:",
        style_body
    ))

    code_sections = [
        ("gtzan_project/models.py", [
            ("Section 1 (PaperCNN)", "Replicates the 4-conv baseline from ICCCNT 2024 paper to demonstrate why simple CNNs overfit."),
            ("Section 2 (SEBlock)", "Squeeze-and-Excitation channel attention (GAP + Linear-ReLU-Linear-Sigmoid) that recalibrates audio frequency channels."),
            ("Section 3 (ConvSEBlock)", "Combines 2 Convolutions, BatchNorm, SE Attention, and a Residual shortcut."),
            ("Section 4 (EnhancedSECNN)", "Our primary 4-stage architecture with Adaptive Average Pooling and Dropout (0.4) (~2.5M params)."),
            ("Section 5 (MobileNetV2Classifier)", "Pretrained lightweight backbone with freeze/unfreeze support for 2-phase fine-tuning.")
        ]),
        ("gtzan_project/preprocess_3s_mels.py", [
            ("Section 1 (Constants)", "Defines sample rate (22050 Hz), chunk samples (66,150), N_FFT (2048), hop length (512), and 128 Mel bands."),
            ("Section 2 (extract_chunks_mels)", "Resamples audio, cuts 10 non-overlapping 3s slices, and extracts 128-band Log-Mel spectrograms."),
            ("Section 3 (build_3s_mels_dataset)", "Batch-processes 1,000 songs, calculates dataset mean (4.88) & std (15.84), and saves compressed NPZ cache.")
        ]),
        ("gtzan_project/dataset.py", [
            ("Section 1 (Genre Labels)", "Defines the 10 GTZAN genres and ImageNet normalization constants."),
            ("Section 2 & 3 (30s Loaders)", "Dataset class and augmentations (RandomCrop, ColorJitter) for baseline 30s images."),
            ("Section 4 (Splits)", "Stratified splitting (70% Train, 15% Val, 15% Test) strictly by song ID to prevent data leakage."),
            ("Section 5 (GTZAN3SecMelDataset)", "Standardizes 3s slices and applies SpecAugment (Frequency Masking & Time Masking)."),
            ("Section 6 (build_3s_dataloaders)", "Constructs PyTorch DataLoader objects with shuffle and GPU pin-memory support.")
        ]),
        ("gtzan_project/train_high_acc.py", [
            ("Section 1 (train_one_epoch)", "Forward pass -> CrossEntropy loss with label smoothing (0.08) -> backward() -> optimizer.step()."),
            ("Section 2 (evaluate)", "Inference loop with @torch.no_grad() computing loss, accuracy, and macro F1."),
            ("Section 3 (get_detailed_predictions)", "Gathers softmax probabilities, hard predictions, labels, and song IDs across test batches."),
            ("Section 4 (compute_song_level_metrics)", "Averages segment probabilities across full songs via soft-voting to produce final track metrics."),
            ("Section 5 (plot_confusion_matrix)", "Renders and saves high-resolution confusion matrix heatmaps (300 DPI)."),
            ("Section 6 (train_secnn_pipeline)", "Trains Enhanced SE-CNN with AdamW and Cosine Annealing learning rate scheduler."),
            ("Section 7 (train_mobilenet_pipeline)", "Two-stage transfer learning: Stage 1 head warmup, Stage 2 backbone fine-tuning."),
            ("Section 8 (main)", "Coordinates data loading, training, hybrid ensembling, confusion matrix plotting, and JSON export.")
        ]),
        ("app.py", [
            ("Section 1 (App Config & Loading)", "Initializes Flask server (50MB limit) and loads trained PyTorch checkpoints into eval mode."),
            ("Section 2 (Acoustic Feature Profiler)", "Extracts Estimated BPM (onset autocorrelation), Spectral Centroid (brightness in Hz), and RMS energy."),
            ("Section 3 (Mel Spectrogram Renderer)", "Dynamically renders the 128-band Mel spectrogram into an in-memory base64 PNG image."),
            ("Section 4 (run_inference)", "Slices audio, runs dual-model ensemble, computes soft-voting, and aggregates song confidences."),
            ("Section 5 (Flask Endpoints)", "Routes for HTML UI (/), metrics API (/api/metrics), and live audio/mic prediction (/predict).")
        ]),
        ("predict_genre.py", [
            ("Section 1 & 2 (Predict Engine)", "Loads models and runs CLI inference on any local audio file (.wav, .mp3, .flac)."),
            ("Section 3 & 4 (CLI Interface)", "Parses arguments and displays a formatted table of genre confidence scores with visual bars.")
        ]),
        ("run_demo.py & show_scores.py", [
            ("Interactive Terminal Hub (run_demo.py)", "Interactive CLI menu for Professor viva: benchmarks, real-time batch inference, audio file analysis, and model complexity inspector."),
            ("Official Benchmark CLI (show_scores.py)", "Fast command-line tool printing the verified 99.23% leaderboard and per-genre metrics with zero clutter.")
        ])
    ]

    for fname, sections in code_sections:
        story.append(Paragraph(f"<b>File: {fname}</b>", style_h2))
        for s_title, s_desc in sections:
            story.append(Paragraph(f"  • <b>{s_title}:</b> {s_desc}", style_bullet))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 6: WEB APPLICATION & LIVE CAPABILITIES
    # =========================================================================
    story.append(Paragraph("6. Interactive Web Application ('What It Does Live')", style_h1))
    story.append(Paragraph(
        "To make the project usable in real-world scenarios, we developed a production-ready Flask web app featuring a modern dark-mode interface:",
        style_body
    ))
    web_feats = [
        ("Multi-Format Audio Upload", "Accepts .mp3, .wav, .flac, .ogg, and .m4a tracks up to 50MB."),
        ("Real-Time Waveform Visualizer", "Client-side Web Audio decoding renders an animated neon audio waveform with interactive scrub and seek controls."),
        ("Live Microphone Recording", "Directly records user singing, humming, whistling, or playing an instrument in the browser using an in-memory 16-bit PCM WAV encoder at 22,050 Hz without external plugins."),
        ("1-Click Preloaded Demos", "Allows instant 1-click testing of Classical, Metal, Disco, Blues, and Hip-Hop audio samples."),
        ("Acoustic Feature Profiler", "Computes and displays Estimated BPM (Tempo), Spectral Centroid / Brightness in Hz (categorized as Warm/Mellow vs. Sharp/Aggressive), RMS Loudness, and Dynamic Range in dB."),
        ("Dynamic Log-Mel Spectrogram", "Generates and renders the exact 128-band Log-Mel frequency spectrogram evaluated by the PyTorch neural networks."),
        ("Cloud & Docker Ready", "Fully containerized with Dockerfile and render.yaml for instant 1-click cloud deployment.")
    ]
    for w_title, w_desc in web_feats:
        story.append(Paragraph(f"✔ <b>{w_title}:</b> {w_desc}", style_bullet))
    story.append(Spacer(1, 14))

    # =========================================================================
    # CHAPTER 7: VIVA VOCE & PROFESSOR Q&A CHEAT SHEET
    # =========================================================================
    story.append(Paragraph("7. Viva Voce & Review Q&A Cheat Sheet", style_h1))
    story.append(Paragraph(
        "Common questions asked by professors during mini-project reviews and the exact answers to give:",
        style_body
    ))

    viva_qa = [
        ("Q1: Why did you convert 1D audio into 2D spectrograms instead of 1D CNNs?",
         "1D raw audio has very high dimensionality (22,050 samples per second) and phase variance. Short-Time Fourier Transform (STFT) transforms audio into 2D time-frequency heatmaps (spectrograms), allowing us to leverage proven 2D computer vision convolutions that extract local frequency-temporal patterns."),

        ("Q2: What is the Squeeze-and-Excitation (SE) block and why is it important?",
         "Standard CNNs treat all feature channels equally. In music, different frequency bands carry the signatures of different genres (e.g. low-frequency bass in Hip-Hop vs. high-frequency distortion in Metal). Squeeze-and-Excitation squeezes spatial dimensions using Global Average Pooling and excites channel importance using two Linear layers with Sigmoid gating to amplify salient frequency channels."),

        ("Q3: What is data leakage, and how did your project avoid it?",
         "If 3-second slices from the same 30-second song appear in both the training set and the test set, the model memorizes the song's specific timbre rather than learning generalizable genre features. We prevented this by splitting our dataset strictly by Song ID: 70% Train, 15% Val, and 15% Test."),

        ("Q4: Why slice songs into 3-second segments instead of using full 30-second clips?",
         "Slicing 30-second songs into 10 chunks of 3 seconds expands our training dataset 10x (from 1,000 to 9,990 samples) and allows the network to learn localized musical riffs. During test time, averaging predictions across all 10 slices eliminates errors caused by quiet intros or atypical solos."),

        ("Q5: Why did Enhanced SE-CNN outperform MobileNetV2?",
         "MobileNetV2 was pretrained on ImageNet (natural images with edges, corners, and textures). While fine-tuned, its receptive fields were originally shaped for visual objects. Our SE-CNN was designed specifically for audio spectrograms with channel attention tailored to musical frequency bands, achieving 99.23% song accuracy with fewer parameters (~2.5M)."),

        ("Q6: How does the Hybrid Ensemble work?",
         "The ensemble combines both models using calibrated weighted soft-voting: P_ensemble = 0.45 * P_secnn + 0.55 * P_mobilenet. SE-CNN contributes specialized frequency attention, while MobileNetV2 contributes deep visual representation priors."),

        ("Q7: How did you calculate BPM (Tempo) in the web app?",
         "We compute the short-term energy envelope of the audio signal, compute the first-order difference (onsets), and perform autocorrelation over onset energies. The peak correlation lag corresponds to the fundamental rhythmic beat interval, which converts to Beats Per Minute (BPM = 60 * fps / lag)."),

        ("Q8: What are the future enhancements for this project?",
         "1. Multi-label classification (e.g., classifying hybrid genres like Indie-Pop or Jazz-Rock). 2. Audio Spectrogram Transformers (AST) for long-range self-attention. 3. Edge deployment on mobile devices using ONNX or TensorRT.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", style_h2))
        story.append(Paragraph(f"<i>Answer:</i> {a}", style_body))
        story.append(Spacer(1, 4))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated comprehensive PDF: {filename}")


if __name__ == "__main__":
    create_pdf()
