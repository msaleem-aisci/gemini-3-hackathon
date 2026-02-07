import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AgriVision Live", layout="centered")
st.title("🌱 AgriVision Live")

# Setup a clean UI container
view_container = st.empty()
info_container = st.container()

# 1. Check if we already have an image in the "Session State"
print("Beore ",st.session_state)
if 'captured_image' not in st.session_state:
    print("AFTER ",st.session_state)
    st.session_state.captured_image = None

# 2. Only show the camera if no image has been captured yet
if st.session_state.captured_image is None:
    img_file_buffer = st.camera_input("Point at a leaf and capture")
    if img_file_buffer:
        # Load and save to session state so camera disappears
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.session_state.captured_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.rerun() # Refresh to hide camera and show analysis

# 3. If an image exists, show the analysis UI
if st.session_state.captured_image is not None:
    opencv_image = st.session_state.captured_image.copy()
    view_container = st.empty()
    view_container.image(opencv_image, caption="Analyzing...")

    # ADD A "RESET" BUTTON to take a new photo
    if st.button("🔄 Scan Another Leaf"):
        st.session_state.captured_image = None
        st.rerun()

    try:
        # --- YOUR GEMINI LOGIC START ---
        pil_image = Image.fromarray(opencv_image)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[pil_image, "Diagnose plant disease. Return JSON with [ymin, xmin, ymax, xmax]."],
            config=types.GenerateContentConfig(
                system_instruction="Return ONLY JSON: {'disease_name': '...', 'coordinates': [ymin, xmin, ymax, xmax], 'treatment': '...'}",
                response_mime_type="application/json"
            )
        )
        
        res_data = json.loads(response.text)
        
        if "coordinates" in res_data:
            h, w, _ = opencv_image.shape
            ymin, xmin, ymax, xmax = res_data['coordinates']
            start_point = (int(xmin * w / 1000), int(ymin * h / 1000))
            end_point = (int(xmax * w / 1000), int(ymax * h / 1000))
            
            # Draw the box
            cv2.rectangle(opencv_image, start_point, end_point, (57, 255, 20), 8) 
            
            # Update the SAME container (No double images!)
            view_container.image(opencv_image, caption="DIAGNOSIS COMPLETE")
            
            st.success(f"**Disease:** {res_data['disease_name']}")
            st.write(f"**Treatment:** {res_data.get('treatment', 'Consult a specialist.')}")
        # --- YOUR GEMINI LOGIC END ---

    except Exception as e:
        st.error(f"Error: {e}")