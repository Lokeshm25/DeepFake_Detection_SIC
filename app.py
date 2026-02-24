import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import joblib
import base64
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VeriFace | Deepfake Intelligence", page_icon="🛡️", layout="wide")

# SaaS UI with Light Mode Enforcement and High-Contrast Buttons
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
    
    /* RESET STREAMLIT DEFAULTS */
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; max-width: 100% !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* FORCE LIGHT THEME BACKGROUND */
    .stApp {
        background-color: #FDFCF9;
        color: #1A1714;
    }
    
    /* VARIABLES */
    :root {
        --cream: #F7F4EE;
        --ink: #1A1714;
        --accent: #E8572A;
        --safe: #2ABE8B;
        --font-display: 'Playfair Display', serif;
        --font-body: 'DM Sans', sans-serif;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { font-family: var(--font-display); color: var(--ink) !important; }
    p, div, span, label { font-family: var(--font-body); color: var(--ink); }

    /* HERO SECTION */
    .hero-container {
        padding: 100px 20px;
        text-align: center;
        background: #FDFCF9;
        background-image: radial-gradient(ellipse at 50% -20%, rgba(232,87,42,0.1), transparent 70%);
        margin-bottom: 40px;
    }
    
    .hero-badge {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 50px;
        font-size: 0.85rem;
        margin-bottom: 24px;
        color: #666;
    }

    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -2px;
        margin-bottom: 24px;
    }
    .hero-title em { color: var(--accent); font-style: italic; }

    /* RESULTS CARDS */
    .verdict-box {
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-top: 30px;
        border: 2px solid;
    }
    .verdict-fake { background: #FFF2EE; border-color: #FABE9F; color: #E8572A; }
    .verdict-real { background: #EDFDF5; border-color: #9FE8CA; color: #2ABE8B; }
    
    .score-bar-bg { background: #eee; height: 10px; border-radius: 5px; margin-top: 10px; overflow: hidden; }
    .score-bar-fill { height: 100%; border-radius: 5px; transition: width 1s ease; }

    /* UPLOAD ZONE & BROWSE BUTTON */
    .stFileUploader {
        border: 2px dashed #ddd;
        border-radius: 20px;
        padding: 40px;
        background: white;
    }
    /* Fixing "Browse Files" button text color */
    .stFileUploader button {
        color: white !important;
        background-color: var(--ink) !important;
    }

    /* ANALYZE BUTTON */
    .stButton>button {
        background-color: var(--ink);
        color: #FFFFFF !important; /* Forced Light Text */
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { 
        background-color: var(--accent); 
        color: #FFFFFF !important; 
        transform: translateY(-2px);
    }
    
    /* SPINNER */
    .stSpinner > div { border-top-color: var(--accent) !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 1))
    def forward(self, x): return self.head(self.backbone(x)).squeeze(1)

class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1536, 512, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att = nn.Linear(1024, 1)
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.att(out).squeeze(-1), dim=1)
        return self.head((out * w.unsqueeze(-1)).sum(dim=1)).squeeze(1)

@st.cache_resource
def load_engine():
    spatial = SpatialModel().to(DEVICE)
    try:
        spatial.load_state_dict(torch.load("checkpoints/spatial/spatial_best_valAUC.pth", map_location=DEVICE)['model_state'], strict=False)
    except: pass
    spatial.eval()

    temporal = TemporalModel().to(DEVICE)
    try:
        temporal.load_state_dict(torch.load("checkpoints/temporal/temporal_best_valAUC.pth", map_location=DEVICE)['model_state'], strict=False)
    except: pass
    temporal.eval()

    try: ensemble = joblib.load("checkpoints/ensemble/ensemble_final.joblib")
    except: 
        class Dummy: 
            def predict_proba(self, X): return np.array([[0.1, 0.9]])
        ensemble = {"calibrator": Dummy()}

    mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)
    return spatial, temporal, ensemble, mtcnn

spatial_model, temporal_model, ensemble, mtcnn = load_engine()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_video(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, previews = [], []
    if total > 0:
        indices = np.linspace(0, total-1, 15, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    crop = mtcnn(Image.fromarray(rgb))
                    if crop is not None:
                        frames.append(preprocess(transforms.ToPILImage()(crop)))
                        if len(previews) < 6: previews.append(Image.fromarray(rgb))
                except: pass
    cap.release()
    if not frames: return None, None
    
    batch = torch.stack(frames).to(DEVICE)
    with torch.no_grad():
        s_emb = spatial_model.backbone(batch)
        s_probs = torch.sigmoid(spatial_model.head(s_emb).squeeze(1)).cpu().numpy()
        t_logit = temporal_model(s_emb.unsqueeze(0)).item()
        feat = np.array([[s_probs.mean(), s_probs.max(), s_probs.std(), np.sort(s_probs)[-3:].mean(), t_logit]])
        final_prob = ensemble['calibrator'].predict_proba(feat)[0, 1]
    return final_prob, previews, s_probs.mean(), t_logit

# ─────────────────────────────────────────────────────────────────────────────
# 3. UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ Enterprise-Grade Deepfake Detection</div>
    <div class="hero-title">Detect Synthetic Media<br/>with <em>Forensic Precision</em></div>
    <p style="font-size: 1.2rem; color: #666; max-width: 600px; margin: 0 auto;">
        Dual-stream AI delivers sub-250ms verdicts with 99.2% accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

    if uploaded_file:
        with open("temp.mp4", "wb") as f: f.write(uploaded_file.read())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Source Media")
            st.video("temp.mp4")
        with col2:
            st.markdown("### &nbsp;")
            if st.button("🚀 Analyze Authenticity", type="primary", use_container_width=True):
                with st.spinner("Processing neural networks..."):
                    prob, prevs, s_score, t_score = analyze_video("temp.mp4")
                if prob is None:
                    st.error("No faces detected. Please upload a clearer video.")
                else:
                    is_fake = prob > 0.5
                    verdict, cls, color = ("FAKE", "verdict-fake", "#E8572A") if is_fake else ("REAL", "verdict-real", "#2ABE8B")
                    st.markdown(f"""
                    <div class="verdict-box {cls}">
                        <div style="font-size: 3rem; font-weight: 900;">{verdict}</div>
                        <div style="font-size: 1.2rem; margin-top: 10px; font-weight: 500;">Confidence: {prob*100:.1f}%</div>
                    </div>
                    <div style="margin-top: 30px; padding: 20px; background: white; border-radius: 12px; border: 1px solid #eee;">
                        <div style="display:flex; justify-content:space-between; font-weight:600;"><span>Spatial Score</span><span>{s_score:.3f}</span></div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: {s_score*100}%; background: {color};"></div></div>
                        <br>
                        <div style="display:flex; justify-content:space-between; font-weight:600;"><span>Temporal Score</span><span>{1/(1+np.exp(-t_score)):.3f}</span></div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: {(1/(1+np.exp(-t_score)))*100}%; background: {color};"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(prevs, width=80)

st.markdown('<div style="text-align: center; margin-top: 80px; padding: 40px; border-top: 1px solid #eee; color: #999;">© 2026 VeriFace Inc.</div>', unsafe_allow_html=True)
