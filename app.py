import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import gzip
import pickle
import os
import gzip
import pickle

model = None

try:
    with gzip.open("model_compressed.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Model loading failed")
    st.write(e)
    st.stop()
# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ASL Detector", page_icon="🤟", layout="wide")

# ---------- LOAD MODEL (FIXED) ----------
model = None
try:
    if os.path.exists("model_compressed.pkl"):
        with gzip.open("model_compressed.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        st.error("❌ model_compressed.pkl not found!")
        st.stop()
except Exception as e:
    st.error("❌ Model loading failed")
    st.write(e)
    st.stop()

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ---------- UI ----------
st.markdown("""
<style>
body {background-color:#0e0e0e;}
.title {text-align:center;font-size:48px;color:#ffb6c1;font-weight:700;}
.sub {text-align:center;color:#aaa;margin-bottom:25px;}
.card {
    background:#161616;
    padding:20px;
    border-radius:20px;
    border:1px solid #ffb6c1;
    box-shadow:0 0 15px rgba(255,182,193,0.1);
}
.big {
    font-size:90px;
    text-align:center;
    color:#ffb6c1;
    font-weight:700;
}
.small {
    text-align:center;
    color:#aaa;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤟 ASL Sign Language Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Upload hand sign → AI predicts the letter</div>", unsafe_allow_html=True)

# ---------- SESSION ----------
if "word" not in st.session_state:
    st.session_state.word = ""

# ---------- LAYOUT ----------
col1, col2 = st.columns([1,1])

# ---------- LEFT ----------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    file = st.file_uploader("📤 Upload Hand Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            pred = model.predict([landmarks])[0]
            prob = model.predict_proba([landmarks])[0]
            conf = int(max(prob) * 100)

            st.image(img, channels="BGR")

            st.markdown(f"<div class='big'>{pred}</div>", unsafe_allow_html=True)
            st.progress(conf)
            st.markdown(f"<div class='small'>Confidence: {conf}%</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ Add Letter"):
                    st.session_state.word += pred
                    st.rerun()
            with c2:
                if st.button("🗑 Clear"):
                    st.session_state.word = ""
                    st.rerun()
        else:
            st.warning("❌ No hand detected")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT ----------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📝 Word Builder")
    st.markdown(
        f"<div class='big' style='font-size:45px'>{st.session_state.word or '...'}</div>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<center style='color:#555'>Built with ❤️ using MediaPipe + ML</center>", unsafe_allow_html=True)
