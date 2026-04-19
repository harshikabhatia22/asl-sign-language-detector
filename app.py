import streamlit as st
import cv2
import pickle
import gzip
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

with gzip.open('model_compressed.pkl', 'rb') as f:
    model = pickle.load(f)

from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

hands = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
HAND_CONNECTIONS = HAND_CONNECTIONS

st.set_page_config(page_title="ASL Sign Language Detector", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .main { background-color: #0d0d0d; }
    .stApp { background-color: #0d0d0d; }
    h1, h2, h3 { color: #ffb6c1 !important; }
    .glow-title { text-align:center; font-size:44px; font-weight:700; color:#ffb6c1; text-shadow: 0 0 10px rgba(255,182,193,0.3); margin-bottom:5px; }
    .subtitle { text-align:center; color:#666; font-size:15px; margin-bottom:20px; }
    .card { background:#161616; border:1px solid #2a2a2a; border-radius:20px; padding:22px; margin:8px 0; }
    .card-pink { background:#161616; border:1.5px solid #ffb6c1; border-radius:20px; padding:22px; margin:8px 0; box-shadow: 0 0 15px rgba(255,182,193,0.1); }
    .detected-letter { font-size:110px; font-weight:700; color:#ffb6c1; line-height:1; text-align:center; }
    .word-display { font-size:34px; font-weight:600; color:#ffc0cb; letter-spacing:10px; min-height:50px; text-align:center; }
    .label { color:#555; font-size:11px; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px; text-align:center; }
    .stat-item { text-align:center; padding:18px 10px; background:#161616; border-radius:15px; border:1px solid #2a2a2a; }
    .stat-number { font-size:26px; font-weight:700; color:#ffb6c1; }
    .stat-label { font-size:11px; color:#555; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
    .history-letter { display:inline-block; background:#1e1e1e; border:1px solid #2a2a2a; border-radius:8px; padding:4px 12px; margin:3px; color:#ffb6c1; font-size:16px; font-weight:600; }
    .stButton > button { border-radius:12px; font-weight:600; font-size:15px; padding:12px 20px; border:none; width:100%; background:linear-gradient(135deg,#ffb6c1,#ff8fab); color:white; box-shadow:0 3px 12px rgba(255,182,193,0.25); }
    .chart-container { border:1.5px solid #ffb6c1; border-radius:20px; overflow:hidden; box-shadow:0 0 20px rgba(255,182,193,0.1); }
    .about-box { background:#161616; border:1px solid #2a2a2a; border-radius:15px; padding:18px; margin-top:12px; }
    .about-item { color:#555; font-size:13px; padding:7px 0; border-bottom:1px solid #1e1e1e; display:flex; justify-content:space-between; }
    .about-value { color:#ffb6c1; font-weight:500; }
    hr { border-color:#1e1e1e; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='padding:25px 0 15px 0;'>
    <div class='glow-title'>🤟 ASL Sign Language Detector</div>
    <div class='subtitle'>Real-time American Sign Language recognition using AI & Computer Vision</div>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("<div class='stat-item'><div class='stat-number'>94.07%</div><div class='stat-label'>Accuracy</div></div>", unsafe_allow_html=True)
with s2:
    st.markdown("<div class='stat-item'><div class='stat-number'>26</div><div class='stat-label'>Letters</div></div>", unsafe_allow_html=True)
with s3:
    st.markdown("<div class='stat-item'><div class='stat-number'>14K+</div><div class='stat-label'>Training Samples</div></div>", unsafe_allow_html=True)
with s4:
    st.markdown("<div class='stat-item'><div class='stat-number'>21</div><div class='stat-label'>Hand Landmarks</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if 'word' not in st.session_state:
    st.session_state.word = ""
if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📷 Live Detection")

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
                        img, hand_landmarks,
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
                cv2.putText(img, prediction, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 182, 193), 4)
                cv2.putText(img, f"{self.confidence}%", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 143, 171), 2)
            else:
                self.letter = "-"
                self.confidence = 0

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="asl",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if ctx.video_processor:
        letter = ctx.video_processor.letter
        confidence = ctx.video_processor.confidence

        st.markdown(f"""
        <div class='card-pink'>
            <div class='label'>Detected Letter</div>
            <div class='detected-letter'>{letter}</div>
            <div style='text-align:center; color:#444; font-size:12px; margin-top:5px;'>
                Confidence &nbsp;<span style='color:#ffb6c1; font-weight:600;'>{confidence}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence) if confidence else 0)

        st.markdown(f"""
        <div class='card'>
            <div class='label'>Word Builder</div>
            <div class='word-display'>{st.session_state.word if st.session_state.word else "· · ·"}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("➕ Add Letter", use_container_width=True):
                if letter != "-":
                    st.session_state.word += letter
                    st.session_state.history.append(letter)
                    st.session_state.history = st.session_state.history[-8:]
                    st.rerun()
        with c2:
            if st.button("⌫ Delete", use_container_width=True):
                st.session_state.word = st.session_state.word[:-1]
                st.rerun()
        with c3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.word = ""
                st.session_state.history = []
                st.rerun()

        if st.session_state.history:
            history_html = "".join([f"<span class='history-letter'>{l}</span>" for l in st.session_state.history])
            st.markdown(f"<div class='card' style='margin-top:8px;'><div class='label'>Recent Letters</div><div style='text-align:center; margin-top:8px;'>{history_html}</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='about-box'>
            <div class='about-item'><span>Algorithm</span><span class='about-value'>Random Forest Classifier</span></div>
            <div class='about-item'><span>Framework</span><span class='about-value'>MediaPipe + OpenCV</span></div>
            <div class='about-item'><span>Dataset</span><span class='about-value'>ASL Alphabet — Kaggle</span></div>
            <div class='about-item' style='border:none;'><span>Hand Landmarks</span><span class='about-value'>21 Keypoints per hand</span></div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📖 ASL Reference Chart")
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.image("asl_chart.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#2a2a2a; font-size:12px; padding:15px;'>
    🤟 ASL Sign Language Detector &nbsp;|&nbsp; MediaPipe + Random Forest + Streamlit
</div>
""", unsafe_allow_html=True)
