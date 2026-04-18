"""
DeepShield Detect — Interface Gradio v3
Clean light mode | English | Creative design
"""

import os, sys, warnings
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gradcam_utils import get_gradcam_heatmap, overlay_gradcam

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'deepshield_best.keras')
# IMG_SIZE lu depuis config.json (généré par Notebook 1)
import json as _json
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "config.json")
with open(_cfg_path) as _f:
    _cfg = _json.load(_f)
IMG_SIZE = _cfg["IMG_SIZE"]
THRESHOLD  = 0.65

# ══════════════════════════════════════════════════════════
# CSS — Clean Editorial Light
# ══════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .main {
    background: #F7F5F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1C1917 !important;
}

/* ── Hero Header ── */
#hero {
    background: #1C1917;
    border-radius: 20px;
    padding: 48px 40px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
#hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,87,51,0.18) 0%, transparent 70%);
    pointer-events: none;
}
#hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    pointer-events: none;
}
#hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #CBD5E0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72em;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 18px;
}
#hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3em !important;
    font-weight: 800 !important;
    color: #FAFAF9 !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em;
}
#hero h1 span { color: #FF5733; }
#hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05em;
    font-weight: 300;
    font-style: italic;
    color: #A8A29E;
    margin: 0 0 20px 0;
}
#hero-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95em;
    color: #D6D3D1;
    line-height: 1.7;
    max-width: 620px;
    margin: 0;
}
#hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 28px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.hero-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6em;
    font-weight: 700;
    color: #FF5733;
}
.hero-stat-label {
    font-size: 0.75em;
    color: #78716C;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Section Labels ── */
.sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68em;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A8A29E;
    margin: 14px 0 6px 0;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
}

/* ── Analyze Button ── */
#analyze-btn {
    background: #1C1917 !important;
    color: #FAFAF9 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95em !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 14px !important;
    width: 100% !important;
    margin-top: 12px !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
}
#analyze-btn:hover {
    background: #FF5733 !important;
    transform: translateY(-1px) !important;
}

/* ── Verdict Box ── */
#verdict-box textarea {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.15em !important;
    font-weight: 700 !important;
    text-align: center !important;
    line-height: 1.7 !important;
    background: #FAFAF9 !important;
    border: 1px solid #E7E5E4 !important;
    border-radius: 12px !important;
    color: #1C1917 !important;
}

/* ── Textboxes ── */
textarea, .gr-textbox textarea {
    background: #FAFAF9 !important;
    border: 1px solid #E7E5E4 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1C1917 !important;
    font-size: 0.9em !important;
}

/* ── Sliders ── */
.gradio-slider label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85em !important;
    font-weight: 500 !important;
    color: #57534E !important;
}

/* ── Accordion ── */
.gr-accordion {
    background: #FFFFFF !important;
    border: 1px solid #E7E5E4 !important;
    border-radius: 16px !important;
}
.gr-accordion .label-wrap span {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #1C1917 !important;
}

/* ── How it works steps ── */
.step-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 12px 0;
}
.step-card {
    background: #F7F5F0;
    border: 1px solid #E7E5E4;
    border-radius: 12px;
    padding: 18px 16px;
}
.step-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8em;
    font-weight: 800;
    color: #E7E5E4;
    line-height: 1;
    margin-bottom: 8px;
}
.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85em;
    font-weight: 700;
    color: #1C1917;
    margin-bottom: 6px;
}
.step-text {
    font-size: 0.82em;
    color: #78716C;
    line-height: 1.5;
}

/* ── Footer ── */
#ds-footer {
    text-align: center;
    font-size: 0.78em;
    color: #A8A29E;
    margin-top: 8px;
    padding: 16px 0 4px;
    border-top: 1px solid #E7E5E4;
    font-family: 'DM Sans', sans-serif;
}
#ds-footer a { color: #78716C; text-decoration: none; }

/* ── Image components ── */
.gr-image { border-radius: 12px !important; overflow: hidden; }

/* ── Labels ── */
label, .label-wrap span {
    font-family: 'DM Sans', sans-serif !important;
    color: #57534E !important;
    font-size: 0.85em !important;
}

/* ── Gradio blocks ── */
.block, .gr-box, .gr-form, .gr-panel {
    background: #FFFFFF !important;
    border-color: #E7E5E4 !important;
}
"""

# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════
print("⏳ Loading model...")
_model    = keras.models.load_model(MODEL_PATH)
_detector = MTCNN()
_model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32'), verbose=0)
print("✅ Model ready.")

# ══════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════
def preprocess_image(pil_image):
    img_rgb = np.array(pil_image.convert('RGB'))
    faces   = _detector.detect_faces(img_rgb)
    if not faces:
        face           = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        detected, conf = False, 0.0
    else:
        best           = max(faces, key=lambda x: x['confidence'])
        x, y, w, h     = best['box']
        x, y           = max(0, x), max(0, y)
        face           = cv2.resize(img_rgb[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        detected, conf = True, best['confidence']
    return face.astype('float32') / 255.0, Image.fromarray(face), detected, conf


def run_prediction(pil_image):
    if pil_image is None:
        return None, None, "← Upload an image to get started", "", "", 0.0, 0.0

    face_norm, face_pil, detected, conf = preprocess_image(pil_image)
    img_input  = np.expand_dims(face_norm, axis=0)
    pred_score = float(_model.predict(img_input, batch_size=1, verbose=0)[0][0])
    is_fake    = pred_score >= THRESHOLD
    confidence = pred_score if is_fake else (1 - pred_score)

    try:
        heatmap     = get_gradcam_heatmap(_model, img_input)
        overlay_arr = overlay_gradcam(face_norm, heatmap, alpha=0.45)
        gradcam_pil = Image.fromarray((overlay_arr * 255).astype(np.uint8))
    except Exception as e:
        print(f"⚠️ Grad-CAM skipped: {e}")
        gradcam_pil = face_pil

    verdict = (
        f"🚨  DEEPFAKE DETECTED\nConfidence: {confidence:.1%}"
        if is_fake else
        f"✅  AUTHENTIC IMAGE\nConfidence: {confidence:.1%}"
    )
    detection_info = (
        f"✅ Face detected  —  MTCNN confidence: {conf:.1%}"
        if detected else
        "⚠️  No face detected — full image analyzed"
    )
    scores_text = (
        f"REAL score  :  {1-pred_score:.4f}   ({1-pred_score:.1%})\n"
        f"FAKE score  :  {pred_score:.4f}   ({pred_score:.1%})\n"
        f"Threshold   :  {THRESHOLD}"
    )
    return face_pil, gradcam_pil, verdict, detection_info, scores_text, 1-pred_score, pred_score


# ══════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════
def build_app():
    with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="DeepShield Detect") as demo:

        # ── Hero ─────────────────────────────────────────
        gr.HTML("""
        <div id="hero">
            <div id="hero-badge">AI-Powered Deepfake Detection</div>
            <h1>Deep<span>Shield</span> Detect</h1>
            <p id="hero-sub">See through the illusion.</p>
            <p id="hero-desc">
                In a world where AI can synthesize convincing faces in seconds,
                knowing what's real matters more than ever.
                DeepShield uses a deep neural network trained on 140,000 real and
                AI-generated faces to expose deepfakes — and shows you exactly
                <em>where</em> in the image the manipulation was detected.
            </p>
            <div id="hero-stats">
                <div>
                    <div class="hero-stat-num">140k</div>
                    <div class="hero-stat-label">Training images</div>
                </div>
                <div>
                    <div class="hero-stat-num">128px</div>
                    <div class="hero-stat-label">Input resolution</div>
                </div>
                <div>
                    <div class="hero-stat-num">Grad-CAM</div>
                    <div class="hero-stat-label">Explainability</div>
                </div>
                <div>
                    <div class="hero-stat-num">MobileNetV2</div>
                    <div class="hero-stat-label">Architecture</div>
                </div>
            </div>
        </div>
        """)

        # ── Main 3-column layout ─────────────────────────
        with gr.Row(equal_height=False):

            # COL 1 — Upload
            with gr.Column(scale=1):
                gr.HTML("<div class='sec-label'>📁 Input Image</div>")
                input_img   = gr.Image(type="pil", label="", height=280, show_label=False)
                analyze_btn = gr.Button("Analyze Image →", elem_id="analyze-btn")
                gr.HTML("""
                <p style="font-size:0.78em; color:#A8A29E; margin-top:10px; line-height:1.6;">
                    Upload any portrait or face photo.<br>
                    The model will automatically detect the face,
                    crop it, and run the deepfake analysis.
                </p>
                """)

            # COL 2 — Visuals
            with gr.Column(scale=1):
                gr.HTML("<div class='sec-label'>👤 Detected Face</div>")
                face_out    = gr.Image(label="", height=210, show_label=False)
                gr.HTML("<div class='sec-label'>🔥 Grad-CAM Heatmap</div>")
                gradcam_out = gr.Image(label="", height=210, show_label=False)

            # COL 3 — Results
            with gr.Column(scale=1):
                gr.HTML("<div class='sec-label'>🎯 Verdict</div>")
                verdict_out = gr.Textbox(
                    label="", lines=3, show_label=False,
                    elem_id="verdict-box",
                    value="← Upload an image to get started"
                )
                gr.HTML("<div class='sec-label'>📡 Face Detection</div>")
                mtcnn_out   = gr.Textbox(label="", lines=2, show_label=False)
                gr.HTML("<div class='sec-label'>📊 Probability Scores</div>")
                scores_out  = gr.Textbox(label="", lines=4, show_label=False)

        # ── Probability bars ─────────────────────────────
        gr.HTML("<div class='sec-label' style='margin-top:16px'>📈 Confidence</div>")
        with gr.Row():
            real_bar = gr.Slider(0, 1, 0, label="🟢 Real probability",  interactive=False)
            fake_bar = gr.Slider(0, 1, 0, label="🔴 Fake probability",  interactive=False)

        # ── How it works ─────────────────────────────────
        with gr.Accordion("ℹ️  How DeepShield works", open=False):
            gr.HTML("""
            <div class="step-grid">
                <div class="step-card">
                    <div class="step-num">01</div>
                    <div class="step-title">Face Detection</div>
                    <div class="step-text">
                        MTCNN (Multi-task Cascaded CNN) scans the image,
                        locates the face with sub-pixel precision, and
                        crops it to a clean 128×128 portrait.
                    </div>
                </div>
                <div class="step-card">
                    <div class="step-num">02</div>
                    <div class="step-title">Neural Analysis</div>
                    <div class="step-text">
                        A MobileNetV2 backbone — pretrained on ImageNet and
                        fine-tuned on 140k real & synthetic faces — extracts
                        deep features and computes a manipulation probability.
                    </div>
                </div>
                <div class="step-card">
                    <div class="step-num">03</div>
                    <div class="step-title">Grad-CAM Explainability</div>
                    <div class="step-text">
                        Gradient-weighted Class Activation Mapping highlights
                        the exact regions that triggered the decision.
                        🔴 Red = suspicious &nbsp; 🔵 Blue = neutral.
                    </div>
                </div>
            </div>
            <p style="font-size:0.82em; color:#78716C; margin:8px 0 4px; font-style:italic;">
                Note: This tool is designed for research and educational purposes.
                Always apply critical judgment when interpreting results.
            </p>
            """)

        # ── Footer ───────────────────────────────────────
        gr.HTML("""
        <div id="ds-footer">
            <strong>DeepShield Detect</strong> &nbsp;·&nbsp;
            Model: MobileNetV2 + Transfer Learning &nbsp;·&nbsp;
            Dataset: 140k Real &amp; Fake Faces (Kaggle) &nbsp;·&nbsp;
            Explainability: Grad-CAM &nbsp;·&nbsp;
            Academic project — EMSI Rabat · 4IASDR
        </div>
        """)

        # ── Event wiring ──────────────────────────────────
        outs = [face_out, gradcam_out, verdict_out, mtcnn_out, scores_out, real_bar, fake_bar]
        analyze_btn.click(fn=run_prediction, inputs=[input_img], outputs=outs)
        input_img.upload(fn=run_prediction,  inputs=[input_img], outputs=outs)

    return demo


# ══════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
