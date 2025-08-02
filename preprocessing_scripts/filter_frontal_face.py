import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def is_frontal_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return False

    # Get landmarks
    landmarks = results.multi_face_landmarks[0].landmark

    # Mediapipe landmark indices
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]

    eye_dist = abs(right_eye.x - left_eye.x)
    nose_centered = abs((left_eye.x + right_eye.x)/2 - nose.x)

    # Same check as before, but with normalized coordinates
    if nose_centered / eye_dist < 0.05:
        return True
    return False

# Paths
input_dir = "faces_aligned_white_bg"
filtered_dir = "frontal_faces"
os.makedirs(filtered_dir, exist_ok=True)

# Filtering
for img_name in tqdm(os.listdir(input_dir)):
    path = os.path.join(input_dir, img_name)
    if is_frontal_face(path):
        shutil.copy(path, os.path.join(filtered_dir, img_name))
