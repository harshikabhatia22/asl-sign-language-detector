import streamlit as st
import cv2
import pickle
import gzip
import numpy as np
import mediapipe as mp

# Load model
with gzip.open('model_compressed.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Page config
st.set_page_config(page_title="ASL Sign Language Detector", page_icon="🤟", layout="wide")

st.title("🤟 ASL Sign Language Detector")
st.markdown("Upload an image of a hand sign and the model will predict the letter.")

# Session state
if 'word' not in st.session_state:
    st.session_state.word = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# File uploader
uploaded_file = st.file_uploader("📤 Upload Hand Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                HAND_CONNECTIONS
            )

        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        proba = model.predict_proba([landmarks])[0]
        prediction = model.predict([landmarks])[0]
        confidence = round(max(proba) * 100, 1)

        st.image(img, channels="BGR", caption="Processed Image")

        st.subheader("🔤 Prediction")
        st.markdown(f"# {prediction}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence}%")

        # Buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("➕ Add Letter"):
                st.session_state.word += prediction
                st.session_state.history.append(prediction)
                st.session_state.history = st.session_state.history[-8:]
                st.rerun()

        with col2:
            if st.button("⌫ Delete"):
                st.session_state.word = st.session_state.word[:-1]
                st.rerun()

        with col3:
            if st.button("🗑 Clear"):
                st.session_state.word = ""
                st.session_state.history = []
                st.rerun()

    else:
        st.warning("❌ No hand detected. Try another image.")

# Word builder UI
st.subheader("📝 Word Builder")
st.write(st.session_state.word if st.session_state.word else "...")

# History
if st.session_state.history:
    st.subheader("🕘 Recent Letters")
    st.write(" ".join(st.session_state.history))

# Footer
st.markdown("---")
st.markdown("🤟 Built using MediaPipe + Machine Learning + Streamlit")