import streamlit as st
import cv2
import pickle
import gzip
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp

# ✅ FIXED MEDIAPIPE IMPORT
mp_hands_module = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model
with gzip.open('model_compressed.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize hands
hands = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
HAND_CONNECTIONS = mp_hands_module.HAND_CONNECTIONS

# Streamlit UI
st.set_page_config(page_title="ASL Sign Language Detector", page_icon="🤟", layout="wide")

st.title("🤟 ASL Sign Language Detector")

# Session state
if 'word' not in st.session_state:
    st.session_state.word = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.letter = "-"
        self.confidence = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 182, 193), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 143, 171), thickness=2)
                )

            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            proba = model.predict_proba([landmarks])[0]
            prediction = model.predict([landmarks])[0]

            self.letter = prediction
            self.confidence = round(max(proba) * 100, 1)

            cv2.putText(img, prediction, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 182, 193), 4)
            cv2.putText(img, f"{self.confidence}%", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 143, 171), 2)
        else:
            self.letter = "-"
            self.confidence = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Webcam stream
ctx = webrtc_streamer(
    key="asl",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Output UI
if ctx.video_processor:
    letter = ctx.video_processor.letter
    confidence = ctx.video_processor.confidence

    st.subheader("Detected Letter")
    st.markdown(f"# {letter}")
    st.progress(int(confidence) if confidence else 0)

    st.subheader("Word Builder")
    st.write(st.session_state.word if st.session_state.word else "...")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("➕ Add Letter"):
            if letter != "-":
                st.session_state.word += letter
                st.session_state.history.append(letter)
                st.session_state.history = st.session_state.history[-8:]
                st.rerun()

    with col2:
        if st.button("⌫ Delete"):
            st.session_state.word = st.session_state.word[:-1]
            st.rerun()

    with col3:
        if st.button("🗑️ Clear"):
            st.session_state.word = ""
            st.session_state.history = []
            st.rerun()

    if st.session_state.history:
        st.subheader("Recent Letters")
        st.write(" ".join(st.session_state.history))

# Reference chart
st.subheader("ASL Reference Chart")
st.image("asl_chart.png", use_container_width=True)