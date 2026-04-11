"""
DeepShield Detect — Interface Gradio v2
Détection de deepfakes avec EfficientNet-B4 + Grad-CAM
Fonctionnalités : Dark/Light mode | FR/EN | Interface restructurée
"""

import os, sys, warnings
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gradcam_utils import get_gradcam_heatmap, overlay_gradcam

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'deepshield_best.keras')
IMG_SIZE   = 224
THRESHOLD  = 0.5

# ══════════════════════════════════════════════════════════
# TRADUCTIONS FR / EN
# ══════════════════════════════════════════════════════════
TRANSLATIONS = {
    "FR": {
        "title":          " DeepShield Detect",
        "subtitle":       "Détection de Deepfakes par Deep Learning",
        "description":    "Uploadez une image de visage pour savoir si elle est **réelle** ou **générée par IA**.",
        "upload_label":   " Image à analyser",
        "analyze_btn":    " Analyser",
        "face_label":     " Visage détecté",
        "gradcam_label":  " Zones analysées (Grad-CAM)",
        "verdict_label":  " Verdict",
        "mtcnn_label":    " Détection MTCNN",
        "scores_label":   " Scores détaillés",
        "real_bar":       " Probabilité RÉEL",
        "fake_bar":       " Probabilité FAKE",
        "verdict_fake":   " DEEPFAKE DÉTECTÉ",
        "verdict_real":   " IMAGE RÉELLE",
        "confidence":     "Confiance",
        "score_real":     "Score RÉEL ",
        "score_fake":     "Score FAKE ",
        "threshold":      "Seuil        ",
        "mtcnn_found":    " Visage détecté — confiance MTCNN",
        "mtcnn_notfound": " Aucun visage détecté — image entière utilisée",
        "no_image":       " Aucune image fournie.",
        "how_title":      " Comment ça fonctionne ?",
        "how_content":    """
**1. Détection du visage** — MTCNN localise et recadre automatiquement le visage.  
**2. Analyse IA** — EfficientNet-B4 (entraîné sur 140k images) calcule la probabilité de manipulation.  
**3. Grad-CAM** — Les zones colorées indiquent les régions suspectes détectées par le modèle.  

 **Bleu / Vert** = zones neutres &nbsp;&nbsp;&nbsp;  **Jaune / Rouge** = zones suspectes
        """,
        "footer":         "**Modèle :** EfficientNet-B4 + Transfer Learning &nbsp;|&nbsp; **Dataset :** 140k Real & Fake Faces &nbsp;|&nbsp; **Projet :** EMSI Rabat — 4IASDR",
        "lang_btn":       "🌐 English",
        "theme_dark_btn": "🌙 Dark Mode",
        "theme_light_btn":"☀️ Light Mode",
    },
    "EN": {
        "title":          " DeepShield Detect",
        "subtitle":       "Deepfake Detection with Deep Learning",
        "description":    "Upload a face image to check if it is **real** or **AI-generated**.",
        "upload_label":   " Image to analyze",
        "analyze_btn":    " Analyze",
        "face_label":     " Detected Face",
        "gradcam_label":  " Analyzed Zones (Grad-CAM)",
        "verdict_label":  " Verdict",
        "mtcnn_label":    " MTCNN Detection",
        "scores_label":   " Detailed Scores",
        "real_bar":       " REAL Probability",
        "fake_bar":       " FAKE Probability",
        "verdict_fake":   " DEEPFAKE DETECTED",
        "verdict_real":   " REAL IMAGE",
        "confidence":     "Confidence",
        "score_real":     "REAL Score",
        "score_fake":     "FAKE Score",
        "threshold":      "Threshold",
        "mtcnn_found":    " Face detected — MTCNN confidence",
        "mtcnn_notfound": " No face detected — using full image",
        "no_image":       " No image provided.",
        "how_title":      "How does it work?",
        "how_content":    """
**1. Face Detection** — MTCNN automatically locates and crops the face.  
**2. AI Analysis** — EfficientNet-B4 (trained on 140k images) computes the manipulation probability.  
**3. Grad-CAM** — Colored zones highlight the suspicious regions detected by the model.  

 **Blue / Green** = neutral zones &nbsp;&nbsp;&nbsp;  **Yellow / Red** = suspicious zones
        """,
        "footer":         "**Model:** EfficientNet-B4 + Transfer Learning &nbsp;|&nbsp; **Dataset:** 140k Real & Fake Faces &nbsp;|&nbsp; **Project:** EMSI Rabat — 4IASDR",
        "lang_btn":       "🌐 Français",
        "theme_dark_btn": "🌙 Dark Mode",
        "theme_light_btn":"☀️ Light Mode",
    }
}

# ══════════════════════════════════════════════════════════
# CSS LIGHT / DARK
# ══════════════════════════════════════════════════════════
CSS_BASE = """
#header { border-radius:16px; padding:32px 24px; text-align:center; margin-bottom:4px; }
#header h1 { font-size:2.2em; margin:0 0 4px 0; }
#header h3 { font-weight:400; margin:0 0 8px 0; }
.section-label { font-size:.78em; font-weight:700; letter-spacing:.07em;
                 text-transform:uppercase; margin-bottom:4px; opacity:.7; }
#analyze-btn { font-size:1.05em !important; font-weight:700 !important;
               border-radius:10px !important; width:100%; margin-top:8px; }
#verdict-box textarea { font-size:1.25em !important; font-weight:700 !important;
                        text-align:center !important; }
.top-bar { display:flex; justify-content:flex-end; gap:8px; margin-bottom:6px; }
"""

CSS_LIGHT = CSS_BASE + """
body,.gradio-container{ background:#f4f6fb !important; color:#1a1a2e !important; }
#header { background:linear-gradient(135deg,#1a1a2e,#0f3460); color:white !important; }
#header h1,#header h3,#header p { color:white !important; }
#header h3 { color:#a8d8ff !important; }
"""

CSS_DARK = CSS_BASE + """
body,.gradio-container{ background:#0d0d1a !important; color:#e0e0f0 !important; }
#header { background:linear-gradient(135deg,#0d0d1a,#0f3460); color:white !important; }
#header h1,#header h3,#header p { color:white !important; }
#header h3 { color:#a8d8ff !important; }
label,.label-wrap span { color:#e0e0f0 !important; }
textarea,input[type=text]{ background:#12122a !important; color:#e0e0f0 !important;
                            border-color:#2a2a4a !important; }
.gr-box,.gr-form,.gr-panel { background:#1a1a2e !important; border-color:#2a2a4a !important; }
"""

# ══════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE & DÉTECTEUR
# ══════════════════════════════════════════════════════════
print("Loading model...")
model    = keras.models.load_model(MODEL_PATH)
detector = MTCNN()
print(" Model and MTCNN ready.")

# ══════════════════════════════════════════════════════════
# PIPELINE PRÉDICTION
# ══════════════════════════════════════════════════════════

def preprocess_image(pil_image):
    img_rgb = np.array(pil_image.convert('RGB'))
    faces   = detector.detect_faces(img_rgb)
    if not faces:
        face     = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        detected, conf = False, 0.0
    else:
        best        = max(faces, key=lambda x: x['confidence'])
        x, y, w, h  = best['box']
        x, y        = max(0, x), max(0, y)
        face        = cv2.resize(img_rgb[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        detected, conf = True, best['confidence']
    return face.astype('float32') / 255.0, Image.fromarray(face), detected, conf


def run_prediction(pil_image, lang):
    t = TRANSLATIONS[lang]
    if pil_image is None:
        return None, None, t["no_image"], "", "", 0.0, 0.0

    face_norm, face_pil, detected, conf = preprocess_image(pil_image)
    img_input  = np.expand_dims(face_norm, axis=0)
    pred_score = float(model.predict(img_input, verbose=0)[0][0])
    is_fake    = pred_score >= THRESHOLD
    confidence = pred_score if is_fake else (1 - pred_score)

    # Grad-CAM
    try:
        heatmap     = get_gradcam_heatmap(model, img_input)
        overlay_arr = overlay_gradcam(face_norm, heatmap, alpha=0.45)
        gradcam_pil = Image.fromarray((overlay_arr * 255).astype(np.uint8))
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gradcam_pil = face_pil

    verdict = (
        f"{t['verdict_fake'] if is_fake else t['verdict_real']}\n"
        f"{t['confidence']} : {confidence:.1%}"
    )
    detection_info = (
        f"{t['mtcnn_found']} : {conf:.1%})" if detected else t["mtcnn_notfound"]
    )
    scores_text = (
        f"{t['score_real']}: {1-pred_score:.4f}  ({1-pred_score:.1%})\n"
        f"{t['score_fake']}: {pred_score:.4f}  ({pred_score:.1%})\n"
        f"{t['threshold']}   : {THRESHOLD}"
    )
    return face_pil, gradcam_pil, verdict, detection_info, scores_text, 1-pred_score, pred_score

# ══════════════════════════════════════════════════════════
# CONSTRUCTION DE L'INTERFACE
# ══════════════════════════════════════════════════════════

def build_app(lang="FR", dark=True):
    t   = TRANSLATIONS[lang]
    css = CSS_DARK if dark else CSS_LIGHT

    with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:

        # États persistants
        state_lang = gr.State(lang)
        state_dark = gr.State(dark)

        # ── Barre de contrôle (haut droite) ─────────────
        with gr.Row():
            gr.HTML("<div style='flex:1'></div>")
            lang_btn  = gr.Button(t["lang_btn"],  size="sm", min_width=110)
            theme_btn = gr.Button(
                t["theme_light_btn"] if dark else t["theme_dark_btn"],
                size="sm", min_width=120
            )

        # ── Header ──────────────────────────────────────
        gr.HTML(f"""
        <div id="header">
            <h1>{t['title']}</h1>
            <h3>{t['subtitle']}</h3>
            <p>{t['description']}</p>
        </div>
        """)

        # ── Corps principal ──────────────────────────────
        with gr.Row(equal_height=False):

            # COL 1 — Upload
            with gr.Column(scale=1):
                gr.Markdown(f"<div class='section-label'>{t['upload_label']}</div>")
                input_img   = gr.Image(type="pil", label="", height=290, show_label=False)
                analyze_btn = gr.Button(t["analyze_btn"], variant="primary", elem_id="analyze-btn")

            # COL 2 — Visuels
            with gr.Column(scale=1):
                gr.Markdown(f"<div class='section-label'>{t['face_label']}</div>")
                face_out    = gr.Image(label="", height=220, show_label=False)
                gr.Markdown(f"<div class='section-label'>{t['gradcam_label']}</div>")
                gradcam_out = gr.Image(label="", height=220, show_label=False)

            # COL 3 — Résultats
            with gr.Column(scale=1):
                gr.Markdown(f"<div class='section-label'>{t['verdict_label']}</div>")
                verdict_out = gr.Textbox(label="", lines=3,  show_label=False, elem_id="verdict-box")
                gr.Markdown(f"<div class='section-label'>{t['mtcnn_label']}</div>")
                mtcnn_out   = gr.Textbox(label="", lines=2,  show_label=False)
                gr.Markdown(f"<div class='section-label'>{t['scores_label']}</div>")
                scores_out  = gr.Textbox(label="", lines=4,  show_label=False)

        # ── Jauges ──────────────────────────────────────
        with gr.Row():
            real_bar = gr.Slider(0, 1, 0, label=t["real_bar"], interactive=False)
            fake_bar = gr.Slider(0, 1, 0, label=t["fake_bar"], interactive=False)

        # ── Accordéon explicatif ─────────────────────────
        with gr.Accordion(t["how_title"], open=False):
            gr.Markdown(t["how_content"])

        # ── Footer ──────────────────────────────────────
        gr.Markdown(f"---\n<center>{t['footer']}</center>")

        # ── Connexions ───────────────────────────────────
        outs = [face_out, gradcam_out, verdict_out, mtcnn_out, scores_out, real_bar, fake_bar]

        analyze_btn.click(
            fn=run_prediction,
            inputs=[input_img, state_lang],
            outputs=outs
        )
        input_img.upload(
            fn=run_prediction,
            inputs=[input_img, state_lang],
            outputs=outs
        )

        # Toggle langue → reload
        def toggle_lang(l, d):
            new_lang = "EN" if l == "FR" else "FR"
            return new_lang, d

        lang_btn.click(
            fn=toggle_lang,
            inputs=[state_lang, state_dark],
            outputs=[state_lang, state_dark]
        ).then(fn=None, js="() => setTimeout(() => location.reload(), 200)")

        # Toggle thème → reload
        def toggle_dark(d):
            return not d

        theme_btn.click(
            fn=toggle_dark,
            inputs=[state_dark],
            outputs=[state_dark]
        ).then(fn=None, js="() => setTimeout(() => location.reload(), 200)")

    return demo

# ══════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = build_app(lang="FR", dark=True)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )