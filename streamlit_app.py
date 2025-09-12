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

# --------------------------- Health/Heart animations + Uploader styling ------------------------
st.markdown("""
<style>
  /* Title: heart beat */
  .hero { display:flex; align-items:center; gap:.6rem; margin:.25rem 0 .15rem 0; }
  .heart {
    font-size: 1.9rem; line-height:1;
    animation: beat 1.8s ease-in-out infinite; transform-origin:center;
    filter: drop-shadow(0 6px 14px rgba(255, 73, 109, .28));
  }
  @keyframes beat { 0%{transform:scale(1)} 20%{transform:scale(1.12)} 40%{transform:scale(1)} }

  /* Upload bar → long rounded pill + animated gradient border */
  [data-testid="stFileUploader"] section { padding: 0; }
  [data-testid="stFileUploaderDropzone"]{
    border: 2px solid transparent;
    border-radius: 999px;
    background:
      linear-gradient(#fff5fa,#fff5fa) padding-box,
      linear-gradient(90deg,#ff6fa0,#ff94b8,#ffb6cf,#ff94b8,#ff6fa0) border-box;
    background-size: 100% 100%, 300% 100%;
    animation: borderflow 6s linear infinite;
    box-shadow: 0 8px 22px rgba(255,105,135,.14);
    padding: .65rem 1rem;
    transition: transform .15s ease, box-shadow .2s ease;
  }
  @keyframes borderflow { 0%{background-position:0 0, 0 0} 100%{background-position:0 0, 300% 0} }
  [data-testid="stFileUploaderDropzone"]:hover{
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(255,105,135,.18);
  }
  /* cloud icon tiny float */
  [data-testid="stFileUploaderDropzone"] svg { filter: drop-shadow(0 4px 10px rgba(255,73,109,.25)); }
  [data-testid="stFileUploaderDropzone"] svg path { animation: float 2.2s ease-in-out infinite; }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-2px)} }

  /* Uploaded image fade-in + soft frame */
  .uploaded-img img { 
    border-radius: 16px; border:2px solid #ffc1d9;
    box-shadow: 0 10px 28px rgba(255, 86, 110, .18);
    animation: fadein .35s ease;
  }
  @keyframes fadein { from{opacity:0; transform:scale(.995)} to{opacity:1; transform:scale(1)} }

  /* Primary button: pulse on hover */
  button[kind="primary"]{
    border-radius: 12px !important; padding:.6rem 1.2rem !important; font-weight:700 !important;
    box-shadow: 0 10px 20px rgba(255, 86, 110, .25) !important;
    transition: transform .1s ease;
  }
  button[kind="primary"]:hover{ transform: translateY(-1px); }

  /* Pretty pills in sidebar (สำหรับลิสต์คลาสถ้าต้องการ) */
  .pill { display:inline-block; padding:.22rem .55rem; border-radius:999px;
    background:#ffe6f2; border:1px solid #ffc7dd; color:#6b2340; font-weight:600; margin:0 .25rem .25rem 0;
    font-size:.85rem;
  }

  /* Custom pink progress bar under uploader */
  .pink-progress-wrap{ margin:.5rem 0 0.4rem 0; }
  .pink-progress{ height:10px; border-radius:999px; background:#ffe8f3;
    border:1px solid #ffb6cf; box-shadow: inset 0 1px 3px rgba(0,0,0,.06); overflow:hidden;}
  .pink-progress .fill{ height:100%;
    background: linear-gradient(90deg,#ff6fa0,#ffa3bd,#ffd5e6);
    width:0%; transition: width .08s ease; }
  .pink-progress-label{ font-size:.85rem; color:#6b2340; opacity:.85; margin-bottom:.15rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------- Header -------------------------------------
st.markdown('<div class="hero"><span class="heart">🫀</span>'
            '<h1 style="margin:0;">ECG Classification Web App</h1></div>', unsafe_allow_html=True)
st.caption("Upload ECG scalogram/spectrogram image → select a model → get class probabilities.")

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

# 3) ตั้งค่าคลาส (ยังแก้ได้ แต่ควรตรงกับตอนเทรน)
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

# แสดง progress bar ระหว่างอ่านไฟล์หลังผู้ใช้เลือก (จำลองการอัปโหลดให้รู้สึกต่อเนื่อง)
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
        st.balloons()

        df = pd.DataFrame({"class": class_names, "prob": probs[0]})
        st.bar_chart(df.set_index("class"))
        st.dataframe(df.sort_values("prob", ascending=False).reset_index(drop=True))
else:
    st.info("Upload an ECG image to start.", icon="📤")
