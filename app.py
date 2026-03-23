import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================================
# 1. SETTINGS & CONFIG
# ==========================================
st.set_page_config(
    page_title="IS 2569 — AI Analysis",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* ─── Base ─── */
    :root {
        --ink:       #0B0D17;
        --surface:   #111320;
        --panel:     #161929;
        --border:    rgba(255,255,255,0.07);
        --blue:      #4F8EF7;
        --teal:      #38C9B0;
        --rose:      #E05C7A;
        --amber:     #E8A840;
        --text:      #D6DCF0;
        --muted:     #7A84A6;
        --glow-blue: rgba(79,142,247,0.25);
        --glow-teal: rgba(56,201,176,0.25);
        --glow-rose: rgba(224,92,122,0.25);
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--ink) !important;
        font-family: 'Outfit', sans-serif;
        color: var(--text);
    }

    /* Subtle star-field noise overlay */
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background-image: radial-gradient(circle at 20% 30%, rgba(79,142,247,0.06) 0%, transparent 60%),
                          radial-gradient(circle at 80% 70%, rgba(56,201,176,0.05) 0%, transparent 60%);
        pointer-events: none; z-index: 0;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* Sidebar nav label */
    .stRadio > label {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted) !important;
    }
    .stRadio div[role="radiogroup"] { gap: 4px; }
    .stRadio div[role="radiogroup"] label {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        transition: all 0.2s !important;
        font-size: 0.9rem !important;
    }
    .stRadio div[role="radiogroup"] label:hover {
        border-color: var(--blue) !important;
        background: rgba(79,142,247,0.08) !important;
    }

    /* ─── Typography ─── */
    h1 { font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
         font-size: 2rem !important; border: none !important; margin-bottom: 4px !important; }
    h2 { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; border: none !important; }
    h3 { font-family: 'Outfit', sans-serif !important; font-weight: 500 !important; }

    /* ─── Metric Cards ─── */
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative; overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .metric-card.blue::before  { background: linear-gradient(90deg, var(--blue), transparent); box-shadow: 0 0 20px var(--glow-blue); }
    .metric-card.teal::before  { background: linear-gradient(90deg, var(--teal), transparent); box-shadow: 0 0 20px var(--glow-teal); }
    .metric-card.rose::before  { background: linear-gradient(90deg, var(--rose), transparent); box-shadow: 0 0 20px var(--glow-rose); }
    .metric-card .label  { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-family: 'Space Mono', monospace; }
    .metric-card .value  { font-size: 2.8rem; font-weight: 700; line-height: 1.1; margin: 10px 0 4px; }
    .metric-card .sub    { font-size: 0.8rem; color: var(--muted); }
    .metric-card.blue  .value { color: var(--blue); }
    .metric-card.teal  .value { color: var(--teal); }
    .metric-card.rose  .value { color: var(--rose); }

    /* ─── Content panels ─── */
    .info-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .info-panel h3 { margin-top: 0; font-size: 1rem; letter-spacing: 0.02em; }

    /* ─── Tag chips ─── */
    .tag {
        display: inline-block;
        background: rgba(79,142,247,0.12);
        color: var(--blue);
        border: 1px solid rgba(79,142,247,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }
    .tag.teal { background: rgba(56,201,176,0.12); color: var(--teal); border-color: rgba(56,201,176,0.3); }

    /* ─── Divider ─── */
    hr { border: none !important; height: 1px !important;
         background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
         margin: 28px 0 !important; }

    /* ─── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, #2A3B6E, #1E2A52) !important;
        color: var(--text) !important;
        border: 1px solid rgba(79,142,247,0.4) !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
        letter-spacing: 0.03em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px var(--glow-blue) !important;
        border-color: var(--blue) !important;
        background: linear-gradient(135deg, #2E4280, #233068) !important;
    }

    /* ─── File uploader ─── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(79,142,247,0.35) !important;
        border-radius: 12px !important;
        background: rgba(79,142,247,0.04) !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--blue) !important;
    }

    /* ─── Selectbox ─── */
    .stSelectbox > div > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px var(--glow-blue) !important;
    }
    .stSelectbox label { color: var(--muted) !important; font-size: 0.8rem !important;
                         text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Space Mono', monospace; }

    /* ─── Alerts ─── */
    .stAlert { background: var(--panel) !important; border-radius: 10px !important; }

    /* ─── Spinner ─── */
    .stSpinner > div { border-top-color: var(--blue) !important; }

    /* ─── Image container ─── */
    .img-wrap {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        overflow: hidden;
    }

    /* ─── Result cards ─── */
    .result-safe {
        background: linear-gradient(135deg, rgba(56,201,176,0.12), rgba(56,201,176,0.04));
        border: 1px solid rgba(56,201,176,0.35);
        border-radius: 16px; padding: 36px; text-align: center;
        box-shadow: 0 8px 32px var(--glow-teal); margin-top: 24px;
    }
    .result-danger {
        background: linear-gradient(135deg, rgba(224,92,122,0.12), rgba(224,92,122,0.04));
        border: 1px solid rgba(224,92,122,0.35);
        border-radius: 16px; padding: 36px; text-align: center;
        box-shadow: 0 8px 32px var(--glow-rose); margin-top: 24px;
    }
    .result-safe   .icon { font-size: 3.5rem; }
    .result-danger .icon { font-size: 3.5rem; }
    .result-safe   .title { color: var(--teal); font-size: 1.6rem; font-weight: 700; margin: 12px 0 4px; }
    .result-danger .title { color: var(--rose); font-size: 1.6rem; font-weight: 700; margin: 12px 0 4px; }
    .result-safe   .sub, .result-danger .sub { color: var(--muted); font-size: 0.9rem; }

    /* ─── Age result ─── */
    .age-result {
        background: linear-gradient(135deg, rgba(79,142,247,0.12), rgba(79,142,247,0.04));
        border: 1px solid rgba(79,142,247,0.35);
        border-radius: 16px; padding: 36px; text-align: center;
        box-shadow: 0 8px 32px var(--glow-blue); margin-top: 20px;
    }
    .age-result .num { font-size: 5rem; font-weight: 700; color: var(--blue); line-height: 1; }
    .age-result .label { color: var(--muted); font-size: 0.95rem; margin-top: 8px; }

    /* ─── Section header ─── */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 12px; margin-bottom: 20px;
    }
    .section-header .icon-box {
        width: 36px; height: 36px;
        background: rgba(79,142,247,0.12);
        border: 1px solid rgba(79,142,247,0.2);
        border-radius: 8px; display: flex;
        align-items: center; justify-content: center;
        font-size: 1.1rem;
    }
    .section-header.teal .icon-box { background: rgba(56,201,176,0.12); border-color: rgba(56,201,176,0.2); }

    /* input field */
    input, textarea {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ─── Sidebar logo ─── */
    .sidebar-badge {
        background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(56,201,176,0.1));
        border: 1px solid rgba(79,142,247,0.25);
        border-radius: 12px; padding: 16px; text-align: center;
        margin-top: 8px;
    }

    /* input field search in selectbox - make dropdown items readable */
    [data-baseweb="select"] input { border: none !important; background: transparent !important; }
    [data-baseweb="popover"] * { background: #1A1E30 !important; color: var(--text) !important; }
    [data-baseweb="menu"] li:hover { background: rgba(79,142,247,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data ───────────────────────────────────────────────────────────────
MUSHROOM_MAP = {
    'odor': {
        'a': 'Almond — อัลมอนด์', 'l': 'Anise — โป๊ยกั๊ก', 'p': 'Pungent — กลิ่นฉุน',
        'n': 'None — ไม่มีกลิ่น', 'f': 'Foul — กลิ่นเหม็น', 'm': 'Musty — กลิ่นอับ',
        's': 'Spicy — กลิ่นเครื่องเทศ', 'y': 'Fishy — กลิ่นคาว', 'c': 'Creosote — กลิ่นน้ำมันดิน'
    },
    'gill-color': {
        'k': 'Black — ดำ', 'n': 'Brown — น้ำตาล', 'b': 'Buff — น้ำตาลอ่อน',
        'h': 'Chocolate — ช็อกโกแลต', 'g': 'Gray — เทา', 'r': 'Green — เขียว',
        'o': 'Orange — ส้ม', 'p': 'Pink — ชมพู', 'u': 'Purple — ม่วง',
        'e': 'Red — แดง', 'w': 'White — ขาว', 'y': 'Yellow — เหลือง'
    },
    'spore-print-color': {
        'k': 'Black — ดำ', 'n': 'Brown — น้ำตาล', 'b': 'Buff — น้ำตาลอ่อน',
        'h': 'Chocolate — ช็อกโกแลต', 'r': 'Green — เขียว', 'o': 'Orange — ส้ม',
        'u': 'Purple — ม่วง', 'w': 'White — ขาว', 'y': 'Yellow — เหลือง'
    },
    'population': {
        'a': 'Abundant — หนาแน่นมาก', 'c': 'Clustered — รวมกลุ่ม',
        'n': 'Numerous — จำนวนมาก', 's': 'Scattered — กระจาย',
        'v': 'Several — ปานกลาง', 'y': 'Solitary — ขึ้นโดดเดี่ยว'
    }
}

# ─── Load models ─────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    age_path       = os.path.join(base_dir, "models", "age_model_best.h5")
    mush_path      = os.path.join(base_dir, "models", "mushroom_model.pkl")
    mush_enc_path  = os.path.join(base_dir, "models", "mushroom_encoders.pkl")
    mush_tgt_path  = os.path.join(base_dir, "models", "mushroom_target_encoder.pkl")
    paths = [age_path, mush_path, mush_enc_path, mush_tgt_path]
    if not all(os.path.exists(p) for p in paths):
        return None
    nn  = tf.keras.models.load_model(age_path, compile=False)
    ml  = joblib.load(mush_path)
    enc = joblib.load(mush_enc_path)
    tgt = joblib.load(mush_tgt_path)
    return nn, ml, enc, tgt

models = load_all_models()
if models:
    age_model, mush_model, mush_encoders, mush_target = models
    model_ready = True
else:
    model_ready = False

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-badge">
        <div style="font-size:1.6rem; margin-bottom:6px;">🌌</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#7A84A6; letter-spacing:0.12em; text-transform:uppercase;">Project</div>
        <div style="font-weight:700; font-size:1.05rem; color:#D6DCF0; margin:2px 0;">IS 2569</div>
        <div style="font-size:0.8rem; color:#7A84A6;">AI Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.7rem; color:#7A84A6; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;">NAVIGATE</p>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📘  Neural Network — ทฤษฎี",
         "📸  ทดสอบทายอายุ  (NN)",
         "📙  Machine Learning — ทฤษฎี",
         "🍄  ทดสอบจำแนกเห็ด  (ML)"],
        label_visibility="collapsed"
    )

# ─── Helpers ─────────────────────────────────────────────────────────────
def section(icon, title, accent="blue"):
    st.markdown(f"""
    <div class="section-header {accent}">
        <div class="icon-box">{icon}</div>
        <span style="font-weight:600; font-size:1rem;">{title}</span>
    </div>
    """, unsafe_allow_html=True)

def metric_card(cls, label, value, sub):
    return f"""
    <div class="metric-card {cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """



# ══════════════════════════════════════════════════════════════════════════
# PAGE: Neural Network Theory
# ══════════════════════════════════════════════════════════════════════════
if page == "📘  Neural Network — ทฤษฎี":

    st.markdown("""
    <div style="margin-bottom:4px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#4F8EF7; letter-spacing:0.15em; text-transform:uppercase;">Neural Network</span>
    </div>
    <h1 style="color:#D6DCF0; margin-bottom:4px;">Age Prediction Model</h1>
    <p style="color:#7A84A6; font-size:1rem; margin-bottom:0;">MobileNetV2 · Transfer Learning · Regression</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("blue", "Accuracy", "60.5%", "±5 ปี threshold"), unsafe_allow_html=True)
    c2.markdown(metric_card("rose", "Mean Abs. Error", "7.2 yrs", "Average deviation"), unsafe_allow_html=True)
    c3.markdown(metric_card("teal", "Architecture", "MobileNetV2", "Transfer Learning"), unsafe_allow_html=True)
    c4.markdown(metric_card("blue", "Input Size", "128×128", "RGB pixels"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(79,142,247,0.35);">
            <h3 style="color:#4F8EF7; margin-top:0;">📊 Dataset</h3>
            <ul style="color:#A8B2C6; font-size:0.9rem; line-height:1.9; padding-left:18px; margin:0;">
                <li><strong>ที่มา:</strong> UTKFace Dataset (Kaggle)</li>
                <li><strong>ขนาด:</strong> ภาพใบหน้า 20,000+ รูป (Unstructured)</li>
                <li><strong>Target:</strong> อายุ 0–116 ปี ระบุในชื่อไฟล์</li>
                <li><strong>Filter:</strong> คัดเฉพาะช่วง <span class="tag">10–60 ปี</span> เพื่อลด imbalance</li>
                <li><strong>หลังกรอง:</strong> เหลือประมาณ 14,000 รูป</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(79,142,247,0.35);">
            <h3 style="color:#4F8EF7; margin-top:0;">⚙️ Data Preprocessing</h3>
            <ul style="color:#A8B2C6; font-size:0.9rem; line-height:1.9; padding-left:18px; margin:0 0 10px 0;">
                <li><strong>Imbalance handling:</strong> กรองช่วงอายุที่มีข้อมูลน้อย</li>
                <li><strong>Augmentation:</strong> Random Rotation ±15°, Horizontal Flip, Zoom 10%</li>
                <li><strong>Normalization:</strong> pixel / 255.0 → [0, 1]</li>
                <li><strong>Resize:</strong> ทุกภาพถูก resize เป็น 128 × 128 × 3</li>
                <li><strong>Split:</strong> Train 80% / Validation 10% / Test 10%</li>
            </ul>
            <div style="background:rgba(79,142,247,0.08); border:1px solid rgba(79,142,247,0.2); border-radius:8px; padding:10px 14px; font-size:0.85rem; color:#A8C4F7;">
                💡 Data Augmentation ช่วยให้โมเดล generalize ได้ดีขึ้น ลด overfitting
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(79,142,247,0.35);">
            <h3 style="color:#4F8EF7; margin-top:0;">🧠 ทฤษฎี Neural Network</h3>
            <p style="color:#A8B2C6; font-size:0.9rem; line-height:1.7; margin:0 0 10px 0;">
                Neural Network เลียนแบบการทำงานของเซลล์ประสาทในสมอง ประกอบด้วย <strong>Layers</strong>
                ที่เชื่อมต่อกันผ่าน <strong>Weights (w)</strong> และ <strong>Biases (b)</strong>
            </p>
            <ul style="color:#A8B2C6; font-size:0.85rem; line-height:1.8; padding-left:18px; margin:0 0 10px 0;">
                <li><strong>Forward Pass:</strong> z = Σ(wᵢxᵢ) + b → ผ่าน Activation function → ได้ output</li>
                <li><strong>Activation (Swish):</strong> f(x) = x · σ(x) ให้ gradient ที่ smooth กว่า ReLU</li>
                <li><strong>Loss (MAE):</strong> L = (1/n) Σ |ŷᵢ − yᵢ| วัดความคลาดเคลื่อนเฉลี่ย</li>
                <li><strong>Backpropagation:</strong> คำนวณ gradient ย้อนกลับทีละ layer</li>
                <li><strong>Adam Optimizer:</strong> ปรับ learning rate แบบ adaptive ต่อ parameter</li>
            </ul>
            <div style="background:rgba(79,142,247,0.07); border:1px solid rgba(79,142,247,0.2); border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#A8C4F7;">
                w ← w − η · ∂L/∂w &nbsp;&nbsp; (η = learning rate)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(79,142,247,0.35);">
            <h3 style="color:#4F8EF7; margin-top:0;">🔬 ทฤษฎี CNN และ MobileNetV2</h3>
            <p style="color:#A8B2C6; font-size:0.9rem; line-height:1.7; margin:0 0 10px 0;">
                <strong>CNN (Convolutional Neural Network)</strong> ออกแบบมาเพื่อประมวลผลรูปภาพ
                โดยใช้ <strong>Convolutional Layer</strong> สกัด feature จาก local patterns
            </p>
            <ul style="color:#A8B2C6; font-size:0.85rem; line-height:1.8; padding-left:18px; margin:0 0 10px 0;">
                <li><strong>Conv Layer:</strong> ใช้ filter/kernel เลื่อนทับภาพ สกัด edge, texture, shape</li>
                <li><strong>Pooling:</strong> ลดขนาด feature map (Max/Average Pooling)</li>
                <li><strong>Depthwise Separable Conv:</strong> MobileNetV2 แยก spatial และ channel convolution ลด parameter 8–9x</li>
                <li><strong>Inverted Residual Block:</strong> Expand → Depthwise Conv → Project พร้อม skip connection</li>
                <li><strong>GlobalAvgPooling:</strong> แปลง feature map 4×4×1280 → vector 1280 มิติ</li>
            </ul>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="tag">Depthwise Conv</span>
                <span class="tag">Skip Connection</span>
                <span class="tag">Lightweight</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(79,142,247,0.35);">
            <h3 style="color:#4F8EF7; margin-top:0;">🔁 Transfer Learning</h3>
            <p style="color:#A8B2C6; font-size:0.9rem; line-height:1.7; margin:0 0 10px 0;">
                แทนที่จะ train โมเดลจากศูนย์ เราใช้ weights จาก <strong>ImageNet</strong>
                (1.2M ภาพ, 1000 classes) ที่โมเดลเรียนรู้ feature ทั่วไปมาแล้ว
            </p>
            <ul style="color:#A8B2C6; font-size:0.85rem; line-height:1.8; padding-left:18px; margin:0 0 10px 0;">
                <li><strong>Phase 1 — Feature Extraction:</strong> Freeze base layers ทั้งหมด train เฉพาะ head</li>
                <li><strong>Phase 2 — Fine-tuning:</strong> Unfreeze layer บน 30 layers ของ base train ด้วย lr ต่ำ (1e-5)</li>
                <li><strong>ข้อดี:</strong> ลดเวลา train จาก ~10 ชั่วโมง เหลือ ~1 ชั่วโมง, ต้องการข้อมูลน้อยกว่า</li>
            </ul>
            <div style="background:rgba(79,142,247,0.07); border:1px solid rgba(79,142,247,0.2); border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#A8C4F7;">
                💡 Low-level features (edge, texture) ใช้ร่วมกับ ImageNet ได้ — High-level features (ใบหน้า) fine-tune เพิ่ม
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(79,142,247,0.35);">
            <h3 style="color:#4F8EF7; margin-top:0;">📈 Test Results & Analysis</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;">
                <div style="background:rgba(79,142,247,0.08); border:1px solid rgba(79,142,247,0.2); border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:#4F8EF7; font-weight:700; font-size:1.1rem;">60.5%</div>
                    <div style="color:#7A84A6; font-size:0.75rem;">Accuracy (±5 ปี)</div>
                </div>
                <div style="background:rgba(224,92,122,0.08); border:1px solid rgba(224,92,122,0.2); border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:#E05C7A; font-weight:700; font-size:1.1rem;">7.2 ปี</div>
                    <div style="color:#7A84A6; font-size:0.75rem;">Mean Abs. Error</div>
                </div>
            </div>
            <p style="color:#A8B2C6; font-size:0.85rem; margin:0 0 8px 0; line-height:1.7;">
                โมเดลทำนายได้แม่นยำในช่วงอายุ <strong>25–45 ปี</strong> ที่มีข้อมูลมากที่สุด
                แต่คลาดเคลื่อนสูงกว่าในช่วงอายุขอบ (10–15 ปี และ 55–60 ปี)
            </p>
            <div style="background:rgba(224,92,122,0.07); border:1px solid rgba(224,92,122,0.2); border-radius:8px; padding:10px 14px; font-size:0.85rem; color:#F0A0B4;">
                ⚠️ ปัจจัยที่กระทบ: แสง, มุมกล้อง, การแต่งหน้า และคุณภาพภาพใน dataset
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Age Prediction Demo
# ══════════════════════════════════════════════════════════════════════════
elif page == "📸  ทดสอบทายอายุ  (NN)":

    st.markdown("""
    <div style="margin-bottom:4px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#4F8EF7; letter-spacing:0.15em; text-transform:uppercase;">Demo — Neural Network</span>
    </div>
    <h1 style="color:#D6DCF0; margin-bottom:4px;">Age Prediction</h1>
    <p style="color:#7A84A6; font-size:1rem;">อัปโหลดรูปใบหน้าเพื่อให้ AI ทำนายอายุ</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if not model_ready:
        st.error("⚠️ โมเดลยังไม่พร้อม — กรุณาตรวจสอบไฟล์ใน /models")
    else:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            section("🖼️", "อัปโหลดรูปภาพ")
            uploaded_file = st.file_uploader(
                "รองรับ JPG · JPEG · PNG",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible"
            )
            if uploaded_file:
                img = Image.open(uploaded_file).convert('RGB')
                st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
                st.image(img, use_column_width=True, caption="ภาพที่อัปโหลด")
                st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            section("🎯", "ผลการวิเคราะห์")
            if not uploaded_file:
                st.markdown("""
                <div style="background:var(--panel); border:1px dashed rgba(255,255,255,0.1); border-radius:14px;
                             padding:60px 20px; text-align:center; color:#7A84A6;">
                    <div style="font-size:3rem; margin-bottom:12px; opacity:0.4;">🧠</div>
                    <p style="margin:0; font-size:0.9rem;">รอรับรูปภาพจากด้านซ้าย…</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("🚀  เริ่มวิเคราะห์อายุ", use_container_width=True):
                    with st.spinner("กำลังประมวลผล…"):
                        prep = np.array(img.resize((128, 128))) / 255.0
                        pred = age_model.predict(np.expand_dims(prep, axis=0))
                        predicted_age = int(pred[0][0])

                    st.markdown(f"""
                    <div class="age-result">
                        <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#7A84A6;
                                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">Predicted Age</div>
                        <div class="num">{predicted_age}</div>
                        <div class="label">ปี · Years Old</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("ℹ️ ความแม่นยำของโมเดลอยู่ที่ ~60% (MAE ±7.2 ปี)")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ML Theory
# ══════════════════════════════════════════════════════════════════════════
elif page == "📙  Machine Learning — ทฤษฎี":

    st.markdown("""
    <div style="margin-bottom:4px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#38C9B0; letter-spacing:0.15em; text-transform:uppercase;">Machine Learning</span>
    </div>
    <h1 style="color:#D6DCF0; margin-bottom:4px;">Mushroom Classification</h1>
    <p style="color:#7A84A6; font-size:1rem;">Random Forest · 100 Trees · Binary Classification</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("teal",  "Accuracy",  "98.5%", "Test Set"), unsafe_allow_html=True)
    c2.markdown(metric_card("blue",  "F1-Score",  "0.985", "Weighted avg"), unsafe_allow_html=True)
    c3.markdown(metric_card("teal",  "Precision", "98.7%", "Poisonous class"), unsafe_allow_html=True)
    c4.markdown(metric_card("rose",  "Recall",    "98.3%", "Edible class"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(56,201,176,0.35);">
            <h3 style="color:#38C9B0; margin-top:0;">📊 Dataset</h3>
            <ul style="color:#A8B2C6; font-size:0.9rem; line-height:1.9; padding-left:18px; margin:0;">
                <li><strong>ที่มา:</strong> UCI Mushroom Classification (Kaggle)</li>
                <li><strong>ขนาด:</strong> 8,124 รายการ (Structured Data)</li>
                <li><strong>Features:</strong> 22 คอลัมน์ เช่น odor, spore-color, gill-color</li>
                <li><strong>Class distribution:</strong> edible 51.8% / poisonous 48.2%</li>
                <li><strong>Target:</strong>
                    <span class="tag teal">edible</span> vs
                    <span class="tag" style="background:rgba(224,92,122,0.12);color:#E05C7A;border-color:rgba(224,92,122,0.3);">poisonous</span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(56,201,176,0.35);">
            <h3 style="color:#38C9B0; margin-top:0;">⚙️ Data Preprocessing</h3>
            <ul style="color:#A8B2C6; font-size:0.9rem; line-height:1.9; padding-left:18px; margin:0 0 10px 0;">
                <li><strong>Encoding:</strong> LabelEncoder แปลง categorical → numeric</li>
                <li><strong>Missing values:</strong> feature <code>stalk-root</code> มีค่า '?' จึงถูก drop</li>
                <li><strong>Split:</strong> Train 80% (6,499) / Test 20% (1,625)</li>
                <li><strong>Feature Selection:</strong> เลือก 4 features ที่ feature importance สูงสุด ได้แก่ odor, spore-print-color, gill-color, population</li>
            </ul>
            <div style="background:rgba(56,201,176,0.07); border:1px solid rgba(56,201,176,0.2); border-radius:8px; padding:10px 14px; font-size:0.85rem; color:#80D8C8;">
                💡 การใช้เพียง 4 features แทน 22 ช่วยลด overfitting และเพิ่ม interpretability
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(56,201,176,0.35);">
            <h3 style="color:#38C9B0; margin-top:0;">📐 ทฤษฎี Decision Tree (รากฐาน)</h3>
            <p style="color:#A8B2C6; font-size:0.9rem; line-height:1.7; margin:0 0 10px 0;">
                Decision Tree แบ่งข้อมูลด้วยการเลือก feature ที่ให้ค่า <strong>Information Gain</strong>
                หรือลด <strong>Gini Impurity</strong> สูงสุดในแต่ละ node
            </p>
            <ul style="color:#A8B2C6; font-size:0.85rem; line-height:1.8; padding-left:18px; margin:0;">
                <li><strong>Gini Impurity:</strong> G = 1 − Σ pᵢ² วัดความไม่บริสุทธิ์ของ node</li>
                <li><strong>Information Gain:</strong> IG = H(parent) − Σ w·H(child)</li>
                <li>ยิ่ง IG สูง → feature นั้นแยก class ได้ดีกว่า</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(56,201,176,0.35);">
            <h3 style="color:#38C9B0; margin-top:0;">🌳 Random Forest — ทฤษฎีและหลักการ</h3>
            <p style="color:#A8B2C6; font-size:0.9rem; line-height:1.7; margin:0 0 10px 0;">
                Random Forest เป็น <strong>Ensemble Learning</strong> ที่รวม Decision Trees หลายต้นเข้าด้วยกัน
                ด้วยเทคนิค <strong>Bagging</strong> (Bootstrap Aggregating)
            </p>
            <ul style="color:#A8B2C6; font-size:0.9rem; line-height:1.9; padding-left:18px; margin:0 0 10px 0;">
                <li><strong>Bootstrap Sampling:</strong> สุ่มตัวอย่าง (แบบ with replacement) สร้าง subset ต่างกันให้แต่ละต้นไม้</li>
                <li><strong>Feature Randomness:</strong> แต่ละ split สุ่มเลือก √p features จาก p ทั้งหมด เพื่อลด correlation ระหว่างต้นไม้</li>
                <li><strong>Majority Voting:</strong> รวมผลจาก 100 ต้นไม้ด้วยการ vote เสียงข้างมาก</li>
                <li><strong>n_estimators = 100:</strong> จำนวนต้นไม้; ยิ่งมากยิ่งเสถียร แต่ใช้เวลานานขึ้น</li>
            </ul>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="tag teal">Ensemble</span>
                <span class="tag teal">Bagging</span>
                <span class="tag teal">Binary Classification</span>
                <span class="tag teal">Low Variance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(56,201,176,0.35);">
            <h3 style="color:#38C9B0; margin-top:0;">📊 Feature Importance</h3>
            <p style="color:#A8B2C6; font-size:0.85rem; margin:0 0 10px 0;">วัดจากค่าเฉลี่ย Gini Impurity Decrease ทั่วทุกต้นไม้</p>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#A8B2C6; margin-bottom:3px;">
                        <span>👃 odor</span><span style="color:#38C9B0;">~0.42</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:8px;">
                        <div style="background:linear-gradient(90deg,#38C9B0,#2A9980); width:92%; height:8px; border-radius:4px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#A8B2C6; margin-bottom:3px;">
                        <span>🎨 spore-print-color</span><span style="color:#38C9B0;">~0.31</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:8px;">
                        <div style="background:linear-gradient(90deg,#38C9B0,#2A9980); width:68%; height:8px; border-radius:4px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#A8B2C6; margin-bottom:3px;">
                        <span>🌈 gill-color</span><span style="color:#4F8EF7;">~0.18</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:8px;">
                        <div style="background:linear-gradient(90deg,#4F8EF7,#3A6FD0); width:40%; height:8px; border-radius:4px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#A8B2C6; margin-bottom:3px;">
                        <span>👥 population</span><span style="color:#4F8EF7;">~0.09</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:8px;">
                        <div style="background:linear-gradient(90deg,#4F8EF7,#3A6FD0); width:20%; height:8px; border-radius:4px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-panel" style="border-left:3px solid rgba(56,201,176,0.35);">
            <h3 style="color:#38C9B0; margin-top:0;">📈 Test Results & Analysis</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;">
                <div style="background:rgba(56,201,176,0.08); border:1px solid rgba(56,201,176,0.2); border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:#38C9B0; font-weight:700; font-size:1.1rem;">98.5%</div>
                    <div style="color:#7A84A6; font-size:0.75rem;">Accuracy</div>
                </div>
                <div style="background:rgba(79,142,247,0.08); border:1px solid rgba(79,142,247,0.2); border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:#4F8EF7; font-weight:700; font-size:1.1rem;">~24 errors</div>
                    <div style="color:#7A84A6; font-size:0.75rem;">จาก 1,625 samples</div>
                </div>
            </div>
            <p style="color:#A8B2C6; font-size:0.85rem; margin:0 0 8px 0; line-height:1.7;">
                โมเดลทำผิดพลาดส่วนใหญ่ในกรณีที่เห็ดมีกลิ่น <strong>none</strong> ซึ่งพบทั้งใน edible และ poisonous
                ทำให้ feature odor แยกไม่ออก โมเดลจึงต้องพึ่ง features อื่นแทน
            </p>
            <div style="background:rgba(224,92,122,0.07); border:1px solid rgba(224,92,122,0.2); border-radius:8px; padding:10px 14px; font-size:0.85rem; color:#F0A0B4;">
                ⚠️ ถึงแม้ accuracy จะสูง ในงานจริงไม่ควรใช้เพียง 4 features อย่างเดียว เพราะอาจพลาดเห็ดพิษได้
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Mushroom Demo
# ══════════════════════════════════════════════════════════════════════════
elif page == "🍄  ทดสอบจำแนกเห็ด  (ML)":

    st.markdown("""
    <div style="margin-bottom:4px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#38C9B0; letter-spacing:0.15em; text-transform:uppercase;">Demo — Machine Learning</span>
    </div>
    <h1 style="color:#D6DCF0; margin-bottom:4px;">Mushroom Classifier</h1>
    <p style="color:#7A84A6; font-size:1rem;">ระบุลักษณะเห็ดแล้ว AI จะบอกว่ากินได้หรือมีพิษ</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if not model_ready:
        st.error("⚠️ โมเดลยังไม่พร้อม — กรุณาตรวจสอบไฟล์ใน /models")
    else:
        col_form, col_out = st.columns([1, 1], gap="large")

        with col_form:
            section("🔬", "ลักษณะของเห็ด", "teal")

            st.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.7rem; color:#7A84A6; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">👃 กลิ่น (Odor)</p>', unsafe_allow_html=True)
            o_c = st.selectbox(
                "odor",
                options=list(MUSHROOM_MAP['odor'].values()),
                label_visibility="collapsed",
                key="odor"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.7rem; color:#7A84A6; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">🌈 สีครีบเห็ด (Gill Color)</p>', unsafe_allow_html=True)
            g_c = st.selectbox(
                "gill",
                options=list(MUSHROOM_MAP['gill-color'].values()),
                label_visibility="collapsed",
                key="gill"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.7rem; color:#7A84A6; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">🎨 สีสปอร์ (Spore Print Color)</p>', unsafe_allow_html=True)
            s_c = st.selectbox(
                "spore",
                options=list(MUSHROOM_MAP['spore-print-color'].values()),
                label_visibility="collapsed",
                key="spore"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'Space Mono\',monospace; font-size:0.7rem; color:#7A84A6; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">👥 ลักษณะประชากร (Population)</p>', unsafe_allow_html=True)
            p_c = st.selectbox(
                "population",
                options=list(MUSHROOM_MAP['population'].values()),
                label_visibility="collapsed",
                key="pop"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            analyze = st.button("🔍  วิเคราะห์ผล", use_container_width=True)

        with col_out:
            section("📊", "ผลการจำแนก", "teal")

            if not analyze:
                st.markdown("""
                <div style="background:var(--panel); border:1px dashed rgba(255,255,255,0.1); border-radius:14px;
                             padding:80px 20px; text-align:center; color:#7A84A6; margin-top:16px;">
                    <div style="font-size:3rem; margin-bottom:12px; opacity:0.4;">🍄</div>
                    <p style="margin:0; font-size:0.9rem;">กดปุ่ม "วิเคราะห์ผล" เพื่อดูผลลัพธ์</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("กำลังวิเคราะห์…"):
                    def get_key(val, mapping):
                        return next(k for k, v in mapping.items() if v == val)

                    vec = np.zeros((1, 22))
                    vec[0, 4]  = mush_encoders['odor'].transform([get_key(o_c, MUSHROOM_MAP['odor'])])[0]
                    vec[0, 9]  = mush_encoders['gill-color'].transform([get_key(g_c, MUSHROOM_MAP['gill-color'])])[0]
                    vec[0, 20] = mush_encoders['spore-print-color'].transform([get_key(s_c, MUSHROOM_MAP['spore-print-color'])])[0]
                    vec[0, 21] = mush_encoders['population'].transform([get_key(p_c, MUSHROOM_MAP['population'])])[0]

                    result = mush_model.predict(vec)[0]

                if result == 0:
                    st.markdown("""
                    <div class="result-safe">
                        <div class="icon">✅</div>
                        <div class="title">เห็ดกินได้</div>
                        <div class="sub">Edible Mushroom — ปลอดภัยสำหรับบริโภค</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-danger">
                        <div class="icon">☠️</div>
                        <div class="title">เห็ดมีพิษ</div>
                        <div class="sub">Poisonous Mushroom — อันตราย ห้ามบริโภค</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Summary of inputs
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px 20px;">
                    <p style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#7A84A6; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 12px;">Input Summary</p>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.85rem; color:#A8B2C6;">
                        <div>👃 {o_c}</div>
                        <div>🌈 {g_c}</div>
                        <div>🎨 {s_c}</div>
                        <div>👥 {p_c}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
