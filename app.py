import streamlit as st
import pickle
import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import mediapipe as mp
import os
import random

# --- Load model and label map ---
model = load_model("CNN_90+acc.keras")
with open("int_labels.pkl", "rb") as f:
    label_to_int = pickle.load(f)
int_to_label = {v: k for k, v in label_to_int.items()}

# --- Load CSV for description and benefits ---
pose_data = pd.read_csv("16.csv")
pose_data['Poses'] = pose_data['Poses'].str.strip().str.lower()

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# --- Background ---
def set_background_from_url(image_url):
    background_css = f'''
    <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        .block-container {{
            background: transparent;
        }}
        header, footer {{
            visibility: hidden;
        }}
    </style>
    '''
    st.markdown(background_css, unsafe_allow_html=True)

# --- Extract Keypoints ---
def extract_keypoints(img):
    img = img.resize((256, 256))
    img_array = np.array(img.convert("RGB"))
    results = pose.process(img_array)
    if results.pose_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        return landmarks
    return None

def normalize_landmarks(landmarks):
    def get_center_point(landmarks, left_idx, right_idx):
        left = np.array(landmarks[left_idx][:2])
        right = np.array(landmarks[right_idx][:2])
        return (left + right) * 0.5

    def get_pose_size(landmarks, torso_size_multiplier=2.5):
        hips_center = get_center_point(landmarks, 23, 24)
        shoulders_center = get_center_point(landmarks, 11, 12)
        torso_size = np.linalg.norm(shoulders_center - hips_center)
        pose_center = hips_center
        dists = np.linalg.norm(np.array(landmarks)[:, :2] - pose_center, axis=1)
        max_dist = np.max(dists)
        return max(torso_size * torso_size_multiplier, max_dist)

    pose_center = get_center_point(landmarks, 23, 24)
    landmarks = np.array(landmarks)
    landmarks[:, :2] -= pose_center
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks[:, :2].flatten()

# --- Session State ---
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

def go_to_next_page():
    st.session_state.page = "next"

def go_back():
    st.session_state.page = "upload"
    st.session_state.prediction_result = None

# --- Upload Page ---
if st.session_state.page == "upload":
    set_background_from_url("https://i.postimg.cc/KYQpc48M/PHOTO-2025-04-19-17-15-35.jpg")

    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-family: 'Times New Roman', serif;
                font-size: 32px;
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 4px #000;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="title">YOGA POSE DETECTION</h2>', unsafe_allow_html=True)

    st.markdown("<br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("UPLOAD THE PICTURE", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file

        # Process and predict
        image = Image.open(uploaded_file)
        keypoints = extract_keypoints(image)

        if keypoints:
            embedding = normalize_landmarks(keypoints)
            reshaped = np.array(embedding).reshape((11, 6, 1))
            prediction = model.predict(np.array([reshaped]))
            predicted_label = int_to_label[np.argmax(prediction)]

            pose_row = pose_data[pose_data["Poses"] == predicted_label.lower()]
            if not pose_row.empty:
                description = pose_row["Description"].values[0]
                benefits = pose_row["Benifits"].values[0]
            else:
                description = "Description not found."
                benefits = "Benefits not found."
        else:
            predicted_label = "Pose not detected"
            description = "Could not extract pose landmarks. Try another image."
            benefits = "Make sure the body is clearly visible."

        st.session_state.prediction_result = {
            "label": predicted_label,
            "description": description,
            "benefits": benefits,
            "image": image
        }

        st.button("Upload", on_click=go_to_next_page)

# --- Result Page ---
elif st.session_state.page == "next":
    set_background_from_url("https://i.postimg.cc/cCj4ps3q/PHOTO-2025-04-19-17-42-27.jpg")

    result = st.session_state.prediction_result

    if result:
        # Top Row: Uploaded + Predicted Image
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Uploaded Image")
            st.image(result["image"], use_container_width=True)

        with col2:
           st.markdown("### Actual Pose")

           # Load reference image for predicted pose
           predicted_label = result["label"].lower().replace(" ", "_")  
           reference_path = f"/Users/bhavya/codes/jupyter/Yoga/yoga-16/reference_images/{predicted_label}.jpg"

           if os.path.exists(reference_path):
               reference_img = Image.open(reference_path)
               st.image(reference_img, caption=predicted_label.replace("_", " ").title(), use_container_width=True)
           else:
               st.write("Image not found.")
           

        # Predicted Pose Name (Centered Below)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align:center; font-size:28px; font-weight:bold; font-family: 'Georgia'; color: white; text-shadow: 1px 1px 3px #000;">
                Predicted Pose: {result['label'].replace('_', ' ').title()}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        st.write("")
        # Bottom Row: Description + Benefits
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="label">Description</div>', unsafe_allow_html=True)
            st.text_area("", result["description"], height=150)

        with col4:
            st.markdown('<div class="label">Benefits</div>', unsafe_allow_html=True)
            st.text_area("", result["benefits"], height=150)

        # Style
        st.markdown("""
            <style>
                .label {
                    font-size: 22px;
                    font-weight: bold;
                    margin-top: 10px;
                    color: white;
                    text-shadow: 1px 1px 3px #000;
                }
                .stImage img {
                    border-radius: 12px;
                    box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
                }
            </style>
        """, unsafe_allow_html=True)

    else:
        st.warning("No prediction available.")

    st.button("Back", on_click=go_back)
