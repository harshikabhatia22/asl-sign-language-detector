import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle, gzip, os

st.set_page_config(page_title="ASL Detector", page_icon="🤟", layout="wide")

# ---------- SAFE MODEL LOAD ----------
model = None
try:
    if os.path.exists("model_compressed.pkl"):
        with gzip.open("model_compressed.pkl", "rb") as f:
            model = pickle.load(f)
    elif os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        st.error("❌ Model file not found (model_compressed.pkl / model.pkl).")
        st.stop()
except Exception as e:
    st.error("❌ Model load failed (version mismatch).")
    st.write(str(e))
    st.stop()

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ---------- UI ----------
st.markdown("""
<style>
body {background:#0e0e0e;}
.title {text-align:center;font-size:46px;color:#ffb6c1;font-weight:700;}
.sub {text-align:center;color:#aaa;margin-bottom:20px;}
.card {background:#161616;padding:20px;border-radius:20px;border:1px solid #ffb6c1;}
.big {font-size:80px;text-align:center;color:#ffb6c1;font-weight:700;}
.small {color:#aaa;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤟 ASL Sign Language Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Upload hand image → detect letter instantly</div>", unsafe_allow_html=True)

if "word" not in st.session_state:
    st.session_state.word = ""

col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            pts = []
            for lm in res.multi_hand_landmarks[0].landmark:
                pts.extend([lm.x, lm.y, lm.z])

            pred = model.predict([pts])[0]
            proba = model.predict_proba([pts])[0]
            conf = int(max(proba)*100)

            st.image(img, channels="BGR")
            st.markdown(f"<div class='big'>{pred}</div>", unsafe_allow_html=True)
            st.progress(conf)
            st.markdown(f"<div class='small'>Confidence: {conf}%</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ Add"):
                    st.session_state.word += pred
                    st.rerun()
            with c2:
                if st.button("🗑 Clear"):
                    st.session_state.word = ""
                    st.rerun()
        else:
            st.warning("No hand detected ❌")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Word Builder")
    st.markdown(f"<div class='big' style='font-size:40px'>{st.session_state.word or '...'}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center style='color:#555'>MediaPipe + ML | Stable Deployment</center>", unsafe_allow_html=True)
