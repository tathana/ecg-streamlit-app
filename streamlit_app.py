# streamlit_app.py
from pathlib import Path
import os, io, time
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from prediction import (
    MODEL_REGISTRY, build_model, load_checkpoint_flex,
    predict_probs, topk_from_probs
)

# -------------------------------- Page setup --------------------------------
st.set_page_config(page_title="ECG Classification", page_icon="🫀", layout="centered")

# ---------------------- EFFECT: Heart Pulse (แทนลูกโป่ง) --------------------
def heart_pulse_effect(duration: float = 2.5):
    """แสดงหัวใจใหญ่กลางจอ เต้นตุบๆ แล้วเฟดหาย"""
    ph = st.empty()
    ph.markdown(f"""
    <style>
      .pulse-wrap {{
        position: fixed; inset: 0; display:flex; align-items:center; justify-content:center;
        z-index: 9999; pointer-events: none;
      }}
      .pulse-heart {{
        font-size: 12rem; color: #ff4b8a;
        animation: beat 1s ease-in-out infinite, fadeout {duration}s forwards;
        text-shadow: 0 0 18px rgba(255,75,138,.8), 0 0 34px rgba(255,148,194,.6);
      }}
      @keyframes beat {{
        0%   {{ transform: scale(1);   }}
        25%  {{ transform: scale(1.25);}}
        40%  {{ transform: scale(1);   }}
        60%  {{ transform: scale(1.18);}}
        100% {{ transform: scale(1);   }}
      }}
      @keyframes fadeout {{
        0%   {{ opacity: 1; }}
        70%  {{ opacity: 1; }}
        100% {{ opacity: 0; }}
      }}
    </style>
    <div class="pulse-wrap">
      <div class="pulse-heart">🫀</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(duration)
    ph.empty()

# --------------------------- Tech Heartbeat + ECG Neon Theme -----------------
st.markdown("""
<style>
  :root{
    --pink:#ff4b8a; --pink2:#ff94c2; --cyan:#45f4ff; --vio:#8a7dff;
    --bg:#0a0b15; --bg2:#111325; --card:rgba(17,19,37,.72);
  }

  /* Global background: subtle moving gradient + tiny bits */
  html, body, .block-container{
    background: radial-gradient(1200px 900px at 15% 10%, #0f1230 0%, var(--bg) 60%),
                linear-gradient(120deg, #0b0d1d 0%, var(--bg) 100%) !important;
  }
  .block-container { padding-top: 1.2rem; }

  /* Floating binary dots for "AI tech" feel */
  body:before{
    content:""; position:fixed; inset:0; pointer-events:none; z-index:-1;
    background-image:
      radial-gradient(rgba(255,255,255,.08) 2px, transparent 2px),
      radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px);
    background-position: 0 0, 25px 35px;
    background-size: 45px 45px, 55px 55px;
    animation: floatBits 16s linear infinite;
    opacity:.45;
  }
  @keyframes floatBits { 0%{transform:translateY(0)} 50%{transform:translateY(-10px)} 100%{transform:translateY(0)} }

  /* Sidebar & header */
  section[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid rgba(255,255,255,.04); }
  .stApp > header { background: transparent; }

  /* Title with beating heart + glow ring */
  .hero { display:flex; align-items:center; gap:.8rem; margin:.15rem 0 .4rem 0; }
  .heart {
    font-size: 2.1rem; line-height:1; position:relative; color:var(--pink);
    animation: beat 1.4s ease-in-out infinite; transform-origin:center;
    text-shadow: 0 0 12px rgba(255,75,138,.55), 0 0 28px rgba(255,148,194,.35);
  }
  .heart:after{
    content:""; position:absolute; inset:-8px; border-radius:50%;
    border:2px solid rgba(255,75,138,.35); animation:pulseRing 1.4s ease-out infinite;
  }
  @keyframes beat { 0%{transform:scale(1)} 25%{transform:scale(1.18)} 40%{transform:scale(1)} 60%{transform:scale(1.13)} 100%{transform:scale(1)} }
  @keyframes pulseRing { 0%{transform:scale(.9); opacity:.8} 70%{transform:scale(1.25); opacity:.1} 100%{transform:scale(1.35); opacity:0} }

  /* Neon ECG line (SVG) under title */
  .ecg-wrap{ margin:.3rem 0 1.1rem 0; }
  .ecg{ width:100%; height:74px; }
  .ecg path{
    fill:none; stroke:url(#grad); stroke-width:3;
    filter: drop-shadow(0 0 6px rgba(69,244,255,.6));
    stroke-linejoin:round; stroke-linecap:round;
    stroke-dasharray: 6 10;
    animation: dash 1.8s linear infinite;
  }
  @keyframes dash { to{ stroke-dashoffset: -160; } }

  /* Uploader: cyber-pill with animated border */
  [data-testid="stFileUploader"] section { padding: 0; }
  [data-testid="stFileUploaderDropzone"]{
    border: 2px solid transparent; border-radius: 16px;
    background:
      linear-gradient(var(--card), var(--card)) padding-box,
      linear-gradient(120deg, var(--pink), var(--vio), var(--cyan), var(--pink)) border-box;
    background-size: 100% 100%, 300% 100%;
    animation: borderflow 8s linear infinite;
    box-shadow: 0 14px 34px rgba(69, 244, 255, .08), 0 10px 22px rgba(255,75,138,.12);
    padding: .8rem 1rem; transition: transform .12s ease, box-shadow .2s ease;
  }
  @keyframes borderflow { 0%{background-position:0 0, 0 0} 100%{background-position:0 0, 300% 0} }
  [data-testid="stFileUploaderDropzone"]:hover{ transform: translateY(-1px); }

  /* Uploaded image frame */
  .uploaded-img img { border-radius: 18px; border:1px solid rgba(255,255,255,.06);
    box-shadow: 0 14px 34px rgba(69,244,255,.10), 0 12px 28px rgba(255,75,138,.14); animation: fadein .28s ease; }
  @keyframes fadein { from{opacity:0; transform:translateY(1px)} to{opacity:1; transform:translateY(0)} }

  /* Buttons */
  button[kind="primary"]{
    border-radius: 12px !important; padding:.62rem 1.1rem !important; font-weight:700 !important;
    background: linear-gradient(90deg, var(--pink), var(--vio), var(--cyan)) !important;
    border:0 !important; box-shadow: 0 10px 22px rgba(255,75,138,.22) !important;
  }

  /* Pills for class names */
  .pill{ display:inline-block; padding:.22rem .6rem; border-radius:999px;
    background:rgba(69,244,255,.10); border:1px solid rgba(69,244,255,.25);
    color:#cdefff; font-weight:600; margin:0 .25rem .25rem 0; font-size:.82rem; }

  /* Pink progress */
  .pink-progress-wrap{ margin:.5rem 0 .4rem 0; }
  .pink-progress{ height:10px; border-radius:999px; background:rgba(255,75,138,.12);
    border:1px solid rgba(255,75,138,.35); overflow:hidden;}
  .pink-progress .fill{ height:100%; background: linear-gradient(90deg, var(--pink), var(--pink2));
    width:0%; transition: width .08s ease; }
  .pink-progress-label{ font-size:.85rem; color:#e6e9ff; opacity:.75; margin-bottom:.15rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------- Header -------------------------------------
st.markdown(
    """
    <div class="hero">
      <span class="heart">🫀</span>
      <h1 style="margin:0;">ECG Classification Web App</h1>
    </div>
    <div class="ecg-wrap">
      <svg class="ecg" viewBox="0 0 800 74" preserveAspectRatio="none">
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#45f4ff"/>
            <stop offset="50%"  stop-color="#ff4b8a"/>
            <stop offset="100%" stop-color="#8a7dff"/>
          </linearGradient>
        </defs>
        <!-- Neon ECG line -->
        <path d="M0,37 L120,37 150,37 165,20 180,55 195,37 250,37 270,37 285,12 300,60 315,37 420,37 440,37 455,18 470,58 485,37 600,37 620,37 635,10 650,60 665,37 800,37"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Upload ECG scalogram/spectrogram → pick a model → get AI-powered class probabilities.")

# -------------------------------- Sidebar -----------------------------------
st.sidebar.header("Settings")

# 1) เลือกโมเดล
model_key = st.sidebar.selectbox(
    "Choose model architecture",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
)

# 2) เลือกไฟล์ checkpoint
BASE_DIR = Path(__file__).parent.resolve()
MODEL_FILEMAP = {
    "ConvNeXt-Tiny":      "models/convnext_tiny_fb_in22k_ft_in1k_fold3_BEST.pt",
    "MobileNetV3-Large":  "models/mobilenetv3_large_100_fold1_BEST.pt",
    "InceptionV3":        "models/inception_v3_tf_in1k_fold5_BEST.pt",
    "EfficientNet-B0":    "models/efficientnet_b0_ra_in1k_fold2_BEST.pt",
    "ECA-ResNet50d":      "models/ecaresnet50d_miil_in1k_fold3_BEST.pt",
}
ckpt_display = st.sidebar.selectbox("Checkpoint file", options=list(MODEL_FILEMAP.keys()))
ckpt_path = str((BASE_DIR / MODEL_FILEMAP[ckpt_display]).resolve())

if not os.path.exists(ckpt_path):
    st.sidebar.error("Checkpoint not found in ./models")
    st.stop()

# 3) ตั้งค่าคลาส (ควรตรงกับตอนเทรน)
class_names = st.sidebar.text_input(
    "Class names (comma-separated)",
    value="CD, HYP, MI, NORM, STTC"
).replace(" ", "").split(",")

# optional: แสดงเป็น pills
st.sidebar.markdown("**Class order used**")
st.sidebar.markdown("".join(f"<span class='pill'>{c}</span>" for c in class_names), unsafe_allow_html=True)

num_classes = len(class_names)
img_size = MODEL_REGISTRY[model_key]["img_size"]

# -------------------------------- Load model --------------------------------
@st.cache_resource(show_spinner=True)
def load_model_cached(model_key: str, num_classes: int, ckpt_path: str):
    model = build_model(model_key, num_classes=num_classes)
    load_checkpoint_flex(model, ckpt_path)   # รองรับ state_dict/whole model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

with st.spinner(f"Loading model: {model_key}  |  weights: {ckpt_display} ..."):
    model = load_model_cached(model_key, num_classes, ckpt_path)

st.success("Model ready!", icon="✅")

# ----------------------------- Upload + Progress + Predict -------------------
uploaded = st.file_uploader("Upload an ECG image (.jpg/.png)", type=["jpg","jpeg","png"])

# แสดง progress bar ระหว่างอ่านไฟล์หลังผู้ใช้เลือก (จำลองความต่อเนื่อง)
progress_ph = st.empty()

if uploaded is not None:
    data = uploaded.getvalue()
    total = len(data)

    def render_progress(pct: float, msg: str):
        progress_ph.markdown(
            f"""
            <div class="pink-progress-wrap">
              <div class="pink-progress-label">{msg}</div>
              <div class="pink-progress"><div class="fill" style="width:{pct:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # อ่านเป็นชิ้น ๆ เพื่ออัปเดตความคืบหน้า
    buf = io.BytesIO()
    step = max(total // 60, 1)   # ~60 steps for smooth animation
    for i in range(0, total, step):
        end = min(total, i + step)
        buf.write(data[i:end])
        pct = (end / total) * 100
        render_progress(pct, f"Receiving file… {end/1024:.1f} / {total/1024:.1f} KB")
        time.sleep(0.01)  # ทำให้เห็นแอนิเมชัน

    progress_ph.empty()

    # เปิดภาพจาก buffer
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    st.container().markdown('<div class="uploaded-img">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ปุ่ม Predict พร้อม progress ตอนรอโมเดล
    if st.button("Predict", type="primary"):
        pb = st.progress(0, text="Running model…")
        for i in range(12):
            time.sleep(0.05)
            pb.progress((i+1)/12, text="Running model…")

        probs = predict_probs(model, image, img_size)     # (1, C)
        pb.progress(1.0, text="Done!")
        time.sleep(0.05)
        pb.empty()

        top = topk_from_probs(probs, class_names, k=min(5, num_classes))

        st.subheader("Prediction")
        best_label, best_prob = top[0]
        st.markdown(f"**Top-1:** `{best_label}` — **{best_prob*100:.2f}%**")

        # ใช้เอฟเฟกต์หัวใจเต้นตุบๆ แทนลูกโป่ง
        heart_pulse_effect()

        df = pd.DataFrame({"class": class_names, "prob": probs[0]})
        st.bar_chart(df.set_index("class"))
        st.dataframe(df.sort_values("prob", ascending=False).reset_index(drop=True))
else:
    st.info("Upload an ECG image to start.", icon="📤")
