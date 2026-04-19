import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import gzip
import pickle
import os

st.set_page_config(page_title="ASL Detector", page_icon="🤟", layout="wide")

# ---------- SAFE MODEL LOAD ----------
model = None
model_loaded = False

try:
    if os.path.exists("model_compressed.pkl"):
        with gzip.open("model_compressed.pkl", "rb") as f:
            model = pickle.load(f)
            model_loaded = True
except:
    model_loaded = False

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ---------- UI ----------
st.markdown("""
<style>
body {background:#0e0e0e;}
.title {text-align:center;font-size:48px;color:#ffb6c1;font-weight:700;}
.card {background:#161616;padding:20px;border-radius:20px;border:1px solid #ffb6c1;}
.big {font-size:80px;text-align:center;color:#ffb6c1;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤟 ASL Detector</div>", unsafe_allow_html=True)

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.image(img, channels="BGR")

        if model_loaded:
            try:
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                pred = model.predict([landmarks])[0]
                st.markdown(f"<div class='big'>{pred}</div>", unsafe_allow_html=True)
            except:
                st.warning("Model error — but detection working ✅")
        else:
            st.warning("Model not loaded — but detection working ✅")

    else:
        st.warning("No hand detected ❌")
