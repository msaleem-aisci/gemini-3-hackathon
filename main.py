import streamlit as st
import cv2
import numpy as np
from PIL import Image
from weather import get_weather
from gemini import analyze_plant_with_agent
import json
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AgriVision Agent", layout="centered")
st.title("🌱 AgriVision: Intelligent Crop Doctor")


with st.sidebar:
    st.header("Settings")
    user_city = st.text_input("Your City", value="Sahiwal")

if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None

if st.session_state.captured_image is None:
    img_file_buffer = st.camera_input("Scan your crop")
    
    if img_file_buffer:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.session_state.captured_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.rerun()

else:
    image_spot = st.empty()
    
    image_spot.image(st.session_state.captured_image, caption="Consulting Agent...", width="stretch")
    
    if st.button("Scan New Plant"):
        st.session_state.captured_image = None
        st.rerun()

    with st.spinner(f"Analyzing leaf & Weather in {user_city}..."):
        pil_image = Image.fromarray(st.session_state.captured_image)

        prompt = f"Diagnose disease. User is in {user_city}. Check weather using the tool."
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                tools=[get_weather],
                system_instruction="""
                Return ONLY JSON. 
                1. Identify disease. 
                2. Check weather. If humidity > 80% or rain, warn user in 'treatment'.
                JSON Format: {'disease_name': '...', 'coordinates': [ymin, xmin, ymax, xmax], 'treatment': '...', 'medicine': '...'}
                """,
                response_mime_type="application/json"
            )
        )
        
        result = json.loads(response.text)
        
        if "coordinates" in result:
            annotated_img = st.session_state.captured_image.copy()
            h, w, _ = annotated_img.shape
            
            coords = result.get('coordinates', [0, 0, 0, 0])
            ymin, xmin, ymax, xmax = coords
            
            start = (int(xmin * w / 1000), int(ymin * h / 1000))
            end = (int(xmax * w / 1000), int(ymax * h / 1000))
            
            cv2.rectangle(annotated_img, start, end, (57, 255, 20), 8)
            
            image_spot.image(annotated_img, caption=f"Detected: {result.get('disease_name', 'Unknown')}", width="stretch")
            
            st.success(f"**Diagnosis:** {result.get('disease_name', 'Unknown')}")
            st.warning(f"**Treatment Plan:** {result.get('treatment', 'No specific treatment advice available.')}")
            st.info(f"**Recommended Medicine:** {result.get('medicine', 'Consult a local expert.')}")
            
        elif "error" in result:
            st.error(f"Error: {result['error']}")