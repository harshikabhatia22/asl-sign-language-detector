import streamlit as st
import cv2
import pickle
import gzip
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

# Load model
with gzip.open('model_compressed.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe setup
hands = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Streamlit page config
st.set_page_config(page_title="ASL Sign Language Detector", page_icon="🤟", layout="wide")

# Title
st.title("🤟 ASL Sign Language Detector")

# Session state
if 'word' not in st.session_state:
    st.session_state.word = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.letter = "-"
        self.confidence = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            proba = model.predict_proba([landmarks])[0]
            prediction = model.predict([landmarks])[0]

            self.letter = prediction
            self.confidence = round(max(proba) * 100, 1)

            cv2.putText(img, prediction, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 182, 193), 4)
        else:
            self.letter = "-"
            self.confidence = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 🔥 UPDATED WEBRTC WITH TURN SERVER
ctx = webrtc_streamer(
    key="asl",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }
)

# Display results
if ctx.video_processor:
    letter = ctx.video_processor.letter
    confidence = ctx.video_processor.confidence

    st.subheader("Detected Letter")
    st.write(letter)
    st.write(f"Confidence: {confidence}%")

    if st.button("Add Letter"):
        if letter != "-":
            st.session_state.word += letter

    if st.button("Clear"):
        st.session_state.word = ""

    st.subheader("Word")
    st.write(st.session_state.word)
