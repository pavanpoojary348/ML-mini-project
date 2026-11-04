"""Build a clean, simple, large-font 8-slide presentation for the Music Genre Classification ML Project."""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Initialize Presentation in 16:9 Widescreen
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Modern High-Contrast Palette
COLOR_BG = RGBColor(15, 23, 42)          # Deep Slate #0F172A
COLOR_CARD = RGBColor(30, 41, 59)        # Card Background #1E293B
COLOR_CARD_BORDER = RGBColor(71, 85, 105)# Slate 600 #475569
COLOR_TEXT_MAIN = RGBColor(255, 255, 255)# Pure White #FFFFFF
COLOR_TEXT_MUTED = RGBColor(203, 213, 225)# Slate 300 #CBD5E1
COLOR_CYAN = RGBColor(56, 189, 248)      # Sky Cyan #38BDF8
COLOR_BLUE = RGBColor(99, 102, 241)      # Indigo #6366F1
COLOR_GREEN = RGBColor(52, 211, 153)     # Emerald #34D399
COLOR_PURPLE = RGBColor(192, 132, 252)   # Purple #C084FC
COLOR_YELLOW = RGBColor(251, 191, 36)    # Amber #FBBF24

def set_slide_background(slide):
    """Draw a dark background covering the entire slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_text, slide_num):
    """Clean, high-visibility header."""
    # Category tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(12)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_CYAN

    # Main Slide Title (Large, bold)
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(10.5), Inches(0.65))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(26)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_TEXT_MAIN

    # Slide number pill
    num_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.7), Inches(0.45), Inches(0.9), Inches(0.45))
    num_shape.fill.solid()
    num_shape.fill.fore_color.rgb = COLOR_CARD
    num_shape.line.color.rgb = COLOR_CARD_BORDER
    tf_num = num_shape.text_frame
    p_num = tf_num.paragraphs[0]
    p_num.alignment = PP_ALIGN.CENTER
    p_num.text = f"{slide_num} / 8"
    p_num.font.size = Pt(13)
    p_num.font.bold = True
    p_num.font.color.rgb = COLOR_CYAN

    # Simple Footer
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.08), Inches(11.8), Inches(0.3))
    tf_foot = foot_box.text_frame
    tf_foot.margin_left = tf_foot.margin_top = tf_foot.margin_right = tf_foot.margin_bottom = 0
    p_foot = tf_foot.paragraphs[0]
    p_foot.text = "Music Genre Classification | Deep Learning Mini-Project"
    p_foot.font.size = Pt(10)
    p_foot.font.color.rgb = COLOR_TEXT_MUTED

def add_card(slide, left, top, width, height, title="", border_color=COLOR_CARD_BORDER):
    """Draw a rounded container card with large title."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD
    card.line.color.rgb = border_color
    card.line.width = Pt(1.5)
    
    if title:
        tbox = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), Inches(0.45))
        tf = tbox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = COLOR_CYAN
    return card

# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
slide1 = prs.slides.add_slide(blank_layout)
set_slide_background(slide1)

# Outer Border Card
main_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
main_card.fill.solid()
main_card.fill.fore_color.rgb = COLOR_CARD
main_card.line.color.rgb = COLOR_BLUE
main_card.line.width = Pt(2.0)

# Project Tag
badge = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3), Inches(1.3), Inches(3.6), Inches(0.45))
badge.fill.solid()
badge.fill.fore_color.rgb = RGBColor(37, 99, 235)
badge.line.fill.background()
tf_b = badge.text_frame
p_b = tf_b.paragraphs[0]
p_b.alignment = PP_ALIGN.CENTER
p_b.text = "MACHINE LEARNING MINI-PROJECT"
p_b.font.size = Pt(12)
p_b.font.bold = True
p_b.font.color.rgb = COLOR_TEXT_MAIN

# Main Title (Large & Clear)
title_box = slide1.shapes.add_textbox(Inches(1.3), Inches(2.0), Inches(10.5), Inches(1.5))
tf_main = title_box.text_frame
tf_main.word_wrap = True
tf_main.margin_left = tf_main.margin_top = tf_main.margin_right = tf_main.margin_bottom = 0

p1 = tf_main.paragraphs[0]
p1.text = "AI-Powered Music Genre Classification"
p1.font.size = Pt(38)
p1.font.bold = True
p1.font.color.rgb = COLOR_TEXT_MAIN

p2 = tf_main.add_paragraph()
p2.text = "Deep Learning with Enhanced SE-CNN & MobileNetV2 on GTZAN Dataset"
p2.font.size = Pt(18)
p2.font.color.rgb = COLOR_CYAN
p2.space_before = Pt(8)

# 3 Big Highlight Stat Cards
kpis = [
    ("99.23%", "Song-Level Accuracy", COLOR_GREEN),
    ("10 Genres", "GTZAN Audio Benchmark", COLOR_CYAN),
    ("Real-Time", "Interactive Web App", COLOR_PURPLE)
]
for i, (val, lbl, col) in enumerate(kpis):
    card_kpi = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3 + i * 3.65), Inches(3.65), Inches(3.4), Inches(1.2))
    card_kpi.fill.solid()
    card_kpi.fill.fore_color.rgb = RGBColor(15, 23, 42)
    card_kpi.line.color.rgb = col
    card_kpi.line.width = Pt(1.5)
    
    tf_k = card_kpi.text_frame
    tf_k.word_wrap = True
    p_v = tf_k.paragraphs[0]
    p_v.alignment = PP_ALIGN.CENTER
    p_v.text = val
    p_v.font.size = Pt(28)
    p_v.font.bold = True
    p_v.font.color.rgb = col
    
    p_l = tf_k.add_paragraph()
    p_l.alignment = PP_ALIGN.CENTER
    p_l.text = lbl
    p_l.font.size = Pt(12)
    p_l.font.color.rgb = COLOR_TEXT_MUTED

# Metadata Section (Simple, readable)
meta_box = slide1.shapes.add_textbox(Inches(1.3), Inches(5.15), Inches(10.5), Inches(1.2))
tf_m = meta_box.text_frame
tf_m.word_wrap = True
tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = 0

p_st = tf_m.paragraphs[0]
p_st.text = "Presented By: [Student Name / Team]  |  Department of Computer Science & Engineering"
p_st.font.size = Pt(14)
p_st.font.bold = True
p_st.font.color.rgb = COLOR_TEXT_MAIN

p_gd = tf_m.add_paragraph()
p_gd.text = "Project Guide / Mentor: [Faculty Name]"
p_gd.font.size = Pt(13)
p_gd.font.color.rgb = COLOR_TEXT_MUTED
p_gd.space_before = Pt(6)

# ==============================================================================
# SLIDE 2: PROJECT OBJECTIVE & DATASET (NO MOTIVATION PART, SIMPLE & LARGE)
# ==============================================================================
slide2 = prs.slides.add_slide(blank_layout)
set_slide_background(slide2)
add_header(slide2, "Project Objective & Dataset", "Overview", 2)

# Left Card: Project Goals (Simple & Direct)
add_card(slide2, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.4), "What Does This Project Do?")
tb_l2 = slide2.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_l2 = tb_l2.text_frame
tf_l2.word_wrap = True
tf_l2.margin_left = tf_l2.margin_top = tf_l2.margin_right = tf_l2.margin_bottom = 0

obj_points = [
    ("Automatic Genre Detection:", " Accurately classify songs into musical genres from audio files or live recordings."),
    ("Visual Audio Representation:", " Convert audio waveforms into 2D frequency images (Log-Mel Spectrograms)."),
    ("High-Accuracy Deep Learning:", " Use an Enhanced CNN with Channel Attention (SE) to capture musical timbre."),
    ("User-Friendly Web App:", " Provide an interactive interface with real-time waveform display, live mic input, and audio analysis.")
]
for i, (b, r) in enumerate(obj_points):
    p = tf_l2.paragraphs[0] if i == 0 else tf_l2.add_paragraph()
    p.space_after = Pt(18)
    rb = p.add_run()
    rb.text = "✔ " + b
    rb.font.bold = True
    rb.font.size = Pt(15)
    rb.font.color.rgb = COLOR_CYAN
    
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(14)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# Right Card: The Dataset (GTZAN)
add_card(slide2, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.4), "The GTZAN Benchmark Dataset")
tb_r2 = slide2.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_r2 = tb_r2.text_frame
tf_r2.word_wrap = True
tf_r2.margin_left = tf_r2.margin_top = tf_r2.margin_right = tf_r2.margin_bottom = 0

dataset_points = [
    ("1,000 Total Audio Tracks:", " Standard benchmark dataset containing 100 tracks per genre, 30 seconds each."),
    ("The 10 Target Genres:", "\n  Blues, Classical, Country, Disco, Hip-Hop,\n  Jazz, Metal, Pop, Reggae, Rock."),
    ("Audio Specification:", " Sample rate of 22,050 Hz, 16-bit mono audio."),
    ("Leak-Free Splitting:", " Split strictly by whole song ID (70% Train, 15% Val, 15% Test) so the model never memorizes tracks.")
]
for i, (b, r) in enumerate(dataset_points):
    p = tf_r2.paragraphs[0] if i == 0 else tf_r2.add_paragraph()
    p.space_after = Pt(16)
    rb = p.add_run()
    rb.text = "• " + b
    rb.font.bold = True
    rb.font.size = Pt(15)
    rb.font.color.rgb = COLOR_GREEN
    
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(14)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# ==============================================================================
# SLIDE 3: STEP-BY-STEP PREPROCESSING PIPELINE
# ==============================================================================
slide3 = prs.slides.add_slide(blank_layout)
set_slide_background(slide3)
add_header(slide3, "How Audio is Processed: Step-by-Step", "Preprocessing Pipeline", 3)

# Top Card: Visual Flowchart
add_card(slide3, Inches(0.8), Inches(1.35), Inches(11.733), Inches(2.2), "End-to-End Audio Pipeline")
if os.path.exists("outputs/pipeline_flowchart.png"):
    slide3.shapes.add_picture("outputs/pipeline_flowchart.png", Inches(1.1), Inches(1.78), width=Inches(11.1))

# Bottom Left: 4 Clear Steps
add_card(slide3, Inches(0.8), Inches(3.75), Inches(6.3), Inches(3.1), "The 4 Pipeline Steps")
tb_steps = slide3.shapes.add_textbox(Inches(1.05), Inches(4.25), Inches(5.8), Inches(2.4))
tf_st = tb_steps.text_frame
tf_st.word_wrap = True
tf_st.margin_left = tf_st.margin_top = tf_st.margin_right = tf_st.margin_bottom = 0

steps = [
    ("Step 1 (Resample):", " Standardize all audio to 22,050 Hz mono."),
    ("Step 2 (3s Slicing):", " Split each 30s song into 10 independent 3-second slices (expands data 10x)."),
    ("Step 3 (Mel Spectrogram):", " Convert audio to 2D image using 128 Mel frequency bands."),
    ("Step 4 (Normalize):", " Scale to decibels (dB) and normalize for smooth training.")
]
for i, (b, r) in enumerate(steps):
    p = tf_st.paragraphs[0] if i == 0 else tf_st.add_paragraph()
    p.space_after = Pt(10)
    rb = p.add_run()
    rb.text = b + " "
    rb.font.bold = True
    rb.font.size = Pt(14)
    rb.font.color.rgb = COLOR_CYAN
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(13.5)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# Bottom Right: Spectrogram Image
add_card(slide3, Inches(7.3), Inches(3.75), Inches(5.233), Inches(3.1), "Resulting 2D Spectrogram Image")
if os.path.exists("outputs/sample_mel_spectrogram.png"):
    slide3.shapes.add_picture("outputs/sample_mel_spectrogram.png", Inches(7.5), Inches(4.2), width=Inches(4.8))

# ==============================================================================
# SLIDE 4: THE DEEP LEARNING MODELS
# ==============================================================================
slide4 = prs.slides.add_slide(blank_layout)
set_slide_background(slide4)
add_header(slide4, "Deep Learning Models: What Learns the Music", "Neural Architectures", 4)

# Left Column: Enhanced SE-CNN
add_card(slide4, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.4), "Model 1: Enhanced SE-CNN (Our Main Model)")
tb_m1 = slide4.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_m1 = tb_m1.text_frame
tf_m1.word_wrap = True
tf_m1.margin_left = tf_m1.margin_top = tf_m1.margin_right = tf_m1.margin_bottom = 0

secnn_points = [
    ("4-Stage Convolution Network:", " Progressively extracts low-level beats to high-level genre patterns (32 -> 64 -> 128 -> 256 channels)."),
    ("Channel Attention (SE Block):", " Automatically learns which frequency bands (bass, midrange, treble) matter most for each genre."),
    ("Batch Normalization & Residuals:", " Keeps training fast and stable; prevents vanishing gradients."),
    ("Dropout & Pooling:", " Reduces parameters to ~2.5M and completely prevents overfitting.")
]
for i, (b, r) in enumerate(secnn_points):
    p = tf_m1.paragraphs[0] if i == 0 else tf_m1.add_paragraph()
    p.space_after = Pt(16)
    rb = p.add_run()
    rb.text = "✔ " + b
    rb.font.bold = True
    rb.font.size = Pt(14.5)
    rb.font.color.rgb = COLOR_CYAN
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(13.5)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# Right Column: MobileNetV2
add_card(slide4, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.4), "Model 2: MobileNetV2 (Transfer Learning)")
tb_m2 = slide4.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_m2 = tb_m2.text_frame
tf_m2.word_wrap = True
tf_m2.margin_left = tf_m2.margin_top = tf_m2.margin_right = tf_m2.margin_bottom = 0

mob_points = [
    ("Pretrained Vision Backbone:", " Uses Google's lightweight MobileNetV2 (~3.3M parameters) pretrained on ImageNet."),
    ("Depthwise Convolutions:", " Highly efficient architecture that runs fast on standard CPUs and mobile hardware."),
    ("Two-Phase Fine-Tuning:", " First warm-up the new classifier head, then fine-tune top layers on audio spectrograms."),
    ("Regularized Classification Head:", " Custom 10-genre output layer with Dropout (0.3) for clean probabilities.")
]
for i, (b, r) in enumerate(mob_points):
    p = tf_m2.paragraphs[0] if i == 0 else tf_m2.add_paragraph()
    p.space_after = Pt(16)
    rb = p.add_run()
    rb.text = "✔ " + b
    rb.font.bold = True
    rb.font.size = Pt(14.5)
    rb.font.color.rgb = COLOR_PURPLE
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(13.5)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# ==============================================================================
# SLIDE 5: PREDICTION & ENSEMBLE (HOW DECISIONS ARE MADE)
# ==============================================================================
slide5 = prs.slides.add_slide(blank_layout)
set_slide_background(slide5)
add_header(slide5, "Inference & Ensembling: Making the Final Decision", "Prediction Strategy", 5)

# Left Card: The Ensemble Process
add_card(slide5, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.4), "1. Combining Both Models (Soft-Voting)")
tb_e1 = slide5.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_e1 = tb_e1.text_frame
tf_e1.word_wrap = True
tf_e1.margin_left = tf_e1.margin_top = tf_e1.margin_right = tf_e1.margin_bottom = 0

e1_points = [
    ("Why Combine Models?", " SE-CNN specializes in audio frequency bands, while MobileNetV2 brings broad visual pattern recognition."),
    ("Calibrated Soft-Voting Formula:", "\n  P_ensemble = (0.45 × SE-CNN) + (0.55 × MobileNet)"),
    ("Multi-Segment Voting:", " All 10 segments of a song are scored. The final song prediction is the average across all segments."),
    ("Eliminates Mistakes:", " If one 3-second slice has quiet talking or an unusual solo, the remaining 9 slices ensure the correct overall genre is picked.")
]
for i, (b, r) in enumerate(e1_points):
    p = tf_e1.paragraphs[0] if i == 0 else tf_e1.add_paragraph()
    p.space_after = Pt(16)
    rb = p.add_run()
    rb.text = "• " + b
    rb.font.bold = True
    rb.font.size = Pt(14.5)
    rb.font.color.rgb = COLOR_GREEN if "Formula" in b else COLOR_CYAN
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(13.5)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# Right Card: Why Our Results Are Trustworthy
add_card(slide5, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.4), "2. Reliable, Leak-Free Evaluation")
tb_e2 = slide5.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_e2 = tb_e2.text_frame
tf_e2.word_wrap = True
tf_e2.margin_left = tf_e2.margin_top = tf_e2.margin_right = tf_e2.margin_bottom = 0

e2_points = [
    ("No Data Leakage:", " We split data strictly by full song. Slices from the same song never appear in both training and testing."),
    ("Unseen Test Evaluation:", " All reported results are measured on completely unseen songs that the model never experienced during training."),
    ("Confidence Probability Output:", " The model outputs exact confidence percentages (e.g., 94% Rock, 4% Metal, 2% Blues)."),
    ("Fast Inference Speed:", " Classifies a full 30-second track in under 0.5 seconds on standard CPU.")
]
for i, (b, r) in enumerate(e2_points):
    p = tf_e2.paragraphs[0] if i == 0 else tf_e2.add_paragraph()
    p.space_after = Pt(16)
    rb = p.add_run()
    rb.text = "✔ " + b
    rb.font.bold = True
    rb.font.size = Pt(14.5)
    rb.font.color.rgb = COLOR_YELLOW
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(13.5)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# ==============================================================================
# SLIDE 6: RESULTS & BENCHMARK PERFORMANCE
# ==============================================================================
slide6 = prs.slides.add_slide(blank_layout)
set_slide_background(slide6)
add_header(slide6, "Experimental Results & Accuracy", "Performance", 6)

# Top Left: Big Stat Box
big_box = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(6.4), Inches(1.3))
big_box.fill.solid()
big_box.fill.fore_color.rgb = RGBColor(15, 23, 42)
big_box.line.color.rgb = COLOR_GREEN
big_box.line.width = Pt(2.0)
tf_bb = big_box.text_frame
tf_bb.word_wrap = True
p_bb1 = tf_bb.paragraphs[0]
p_bb1.alignment = PP_ALIGN.CENTER
p_bb1.text = "99.23% SONG ACCURACY"
p_bb1.font.size = Pt(28)
p_bb1.font.bold = True
p_bb1.font.color.rgb = COLOR_GREEN
p_bb2 = tf_bb.add_paragraph()
p_bb2.alignment = PP_ALIGN.CENTER
p_bb2.text = "Achieved by Enhanced SE-CNN on Unseen Test Songs"
p_bb2.font.size = Pt(13)
p_bb2.font.color.rgb = COLOR_TEXT_MAIN

# Middle Left: Clean Comparison Table (Large Font)
add_card(slide6, Inches(0.8), Inches(2.9), Inches(6.4), Inches(2.1), "Model Performance Comparison")
table_shape = slide6.shapes.add_table(4, 4, Inches(0.95), Inches(3.45), Inches(6.1), Inches(1.4))
table = table_shape.table
table.columns[0].width = Inches(2.5)
table.columns[1].width = Inches(1.2)
table.columns[2].width = Inches(1.2)
table.columns[3].width = Inches(1.2)

headers = ["Model", "3s Slices", "Full Song", "F1-Score"]
rows = [
    ["Enhanced SE-CNN", "97.30%", "99.23%", "99.22%"],
    ["Hybrid Ensemble", "96.20%", "97.69%", "97.64%"],
    ["MobileNetV2", "89.49%", "92.28%", "92.24%"]
]

for col_idx, h in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(37, 99, 235)
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_TEXT_MAIN
    p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

for row_idx, r_data in enumerate(rows):
    for col_idx, val in enumerate(r_data):
        cell = table.cell(row_idx + 1, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_CARD if row_idx % 2 == 0 else RGBColor(20, 27, 45)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.size = Pt(12)
        p.font.bold = (row_idx == 0 and col_idx >= 2)
        p.font.color.rgb = COLOR_GREEN if (row_idx == 0 and col_idx >= 2) else COLOR_TEXT_MAIN
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

# Bottom Left: Key Findings
add_card(slide6, Inches(0.8), Inches(5.15), Inches(6.4), Inches(1.65), "Key Findings")
tb_kf = slide6.shapes.add_textbox(Inches(1.0), Inches(5.55), Inches(6.0), Inches(1.1))
tf_kf = tb_kf.text_frame
tf_kf.word_wrap = True
tf_kf.margin_left = tf_kf.margin_top = tf_kf.margin_right = tf_kf.margin_bottom = 0

p_k1 = tf_kf.paragraphs[0]
p_k1.text = "• 100% Accuracy on Blues & Metal genres (zero misclassifications)."
p_k1.font.size = Pt(13)
p_k1.font.bold = True
p_k1.font.color.rgb = COLOR_CYAN

p_k2 = tf_kf.add_paragraph()
p_k2.text = "• Full-track averaging boosted slice accuracy from 97.3% to 99.23%."
p_k2.font.size = Pt(13)
p_k2.font.color.rgb = COLOR_TEXT_MAIN
p_k2.space_before = Pt(4)

# Right Column: Confusion Matrix Picture
add_card(slide6, Inches(7.4), Inches(1.4), Inches(5.133), Inches(5.4), "Song-Level Confusion Matrix")
if os.path.exists("results_gtzan_high_acc/cm_ensemble_song.png"):
    slide6.shapes.add_picture("results_gtzan_high_acc/cm_ensemble_song.png", Inches(7.55), Inches(1.95), width=Inches(4.8))

# ==============================================================================
# SLIDE 7: INTERACTIVE WEB APPLICATION (WHAT IT DOES)
# ==============================================================================
slide7 = prs.slides.add_slide(blank_layout)
set_slide_background(slide7)
add_header(slide7, "Interactive Web Application: What It Does", "Live Deployment", 7)

# 4 Clear Feature Cards with Large Text
app_cards = [
    ("1. File Upload & Waveform Player", [
        ("Upload Any Song:", " Supports MP3, WAV, FLAC, OGG audio files."),
        ("Real-Time Waveform:", " Interactive visual player with seek and pause controls.")
    ], Inches(0.8), Inches(1.4), COLOR_CYAN),

    ("2. Live Microphone Recording", [
        ("Record Direct from Browser:", " Hum, sing, or play an instrument live."),
        ("In-Memory 16-bit WAV:", " Real-time conversion without external plugins.")
    ], Inches(6.8), Inches(1.4), COLOR_GREEN),

    ("3. Audio Trait Profiler", [
        ("Estimated Tempo (BPM):", " Detects rhythm speed and tempo automatically."),
        ("Acoustic Brightness (Hz):", " Classifies tone as Warm/Mellow vs. Sharp/Aggressive.")
    ], Inches(0.8), Inches(4.15), COLOR_PURPLE),

    ("4. Spectrogram & Model Insights", [
        ("Live Mel Spectrogram:", " Renders the exact 128-band frequency image tested."),
        ("1-Click Demo Songs:", " Test Classical, Metal, Disco, Blues, Hip-Hop with one click.")
    ], Inches(6.8), Inches(4.15), COLOR_YELLOW),
]

for title, bullets, left, top, col in app_cards:
    add_card(slide7, left, top, Inches(5.7), Inches(2.65), title, border_color=col)
    tbox = slide7.shapes.add_textbox(left + Inches(0.25), top + Inches(0.65), Inches(5.2), Inches(1.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    for i, (b, r) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        rb = p.add_run()
        rb.text = "✔ " + b + " "
        rb.font.bold = True
        rb.font.size = Pt(14)
        rb.font.color.rgb = col
        rr = p.add_run()
        rr.text = r
        rr.font.size = Pt(13.5)
        rr.font.color.rgb = COLOR_TEXT_MAIN

# ==============================================================================
# SLIDE 8: CONCLUSION & FUTURE SCOPE
# ==============================================================================
slide8 = prs.slides.add_slide(blank_layout)
set_slide_background(slide8)
add_header(slide8, "Conclusion & Future Work", "Summary", 8)

# Left Column: Project Summary
add_card(slide8, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.4), "1. Summary of Achievements")
tb_s1 = slide8.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_s1 = tb_s1.text_frame
tf_s1.word_wrap = True
tf_s1.margin_left = tf_s1.margin_top = tf_s1.margin_right = tf_s1.margin_bottom = 0

achievements = [
    ("99.23% Song Accuracy:", " Outperformed standard baseline CNNs using Squeeze-and-Excitation (SE) channel attention."),
    ("Leak-Free Methodology:", " Proved genuine model generalization with strict track-level train/test splits."),
    ("End-to-End Working System:", " Built a full-stack Flask web application with live mic recording and audio feature analysis."),
    ("Ready for Cloud Deployment:", " Packaged with Docker and Render configurations for easy 1-click hosting.")
]
for i, (b, r) in enumerate(achievements):
    p = tf_s1.paragraphs[0] if i == 0 else tf_s1.add_paragraph()
    p.space_after = Pt(18)
    rb = p.add_run()
    rb.text = "✔ " + b
    rb.font.bold = True
    rb.font.size = Pt(15)
    rb.font.color.rgb = COLOR_GREEN
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(14)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# Right Column: Future Enhancements
add_card(slide8, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.4), "2. Future Enhancements")
tb_s2 = slide8.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.1), Inches(4.5))
tf_s2 = tb_s2.text_frame
tf_s2.word_wrap = True
tf_s2.margin_left = tf_s2.margin_top = tf_s2.margin_right = tf_s2.margin_bottom = 0

future_points = [
    ("Multi-Label Genre Classification:", " Classify songs that belong to multiple genres (e.g. Pop-Rock, Jazz-Hop)."),
    ("Audio Spectrogram Transformers (AST):", " Experiment with attention transformers for capturing full-length song structure."),
    ("Mobile Application Deployment:", " Export lightweight models via ONNX to run directly on Android/iOS smartphones."),
    ("Expanded Datasets:", " Train and evaluate on larger datasets such as the Free Music Archive (FMA).")
]
for i, (b, r) in enumerate(future_points):
    p = tf_s2.paragraphs[0] if i == 0 else tf_s2.add_paragraph()
    p.space_after = Pt(18)
    rb = p.add_run()
    rb.text = "• " + b
    rb.font.bold = True
    rb.font.size = Pt(15)
    rb.font.color.rgb = COLOR_CYAN
    rr = p.add_run()
    rr.text = r
    rr.font.size = Pt(14)
    rr.font.color.rgb = COLOR_TEXT_MAIN

# Save the presentation
output_pptx = "Music_Genre_Classification_Simple.pptx"
prs.save(output_pptx)
print(f"Successfully generated simplified, large-text 8-slide presentation: {output_pptx}")

