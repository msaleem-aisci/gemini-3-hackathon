import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import re
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

st.set_page_config(page_title="AgriVision Pro", layout="centered")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key missing. Please check .env or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)

def get_weather_context(city):
    """Fetches real-time weather to inject into the Agent's brain."""
    w_key = os.getenv("WEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY")
    if not w_key: return "Weather data unavailable."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={w_key}&units=metric"
        data = requests.get(url).json()
        if data.get("cod") != 200: return "Weather unavailable."
        return f"{data['main']['temp']}°C, {data['weather'][0]['description']}, Humidity: {data['main']['humidity']}%"
    except:
        return "Weather service error."

def clean_json_text(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def plant_analyzer(pil_image, city):
    
    weather_fact = get_weather_context(city)
    
    prompt = f"""
    You are an Expert Agronomist Agent. 
    User Location: {city}.
    LIVE CONTEXT -> Current Weather: {weather_fact}
    
    YOUR MISSION:
    1. ANALYZE the image to identify the disease.
    2. LOCATE THE DISEASE: You must identify the specific damaged area or spots.
    3. REASON about the Weather: Look at the weather data above.
    4. SEARCH GOOGLE: Find medicine prices in Pakistan.
    
    OUTPUT:
    Return strictly valid JSON text. Do not wrap in markdown.
    {{
        "disease_name": "str",
        "treatment": "str",
        "medicine": "str",
        "coordinates": [ymin, xmin, ymax, xmax],
        "search_finding": "str"
    }}

    IMPORTANT INSTRUCTION FOR COORDINATES:
    - Do NOT select the entire image.
    - Do NOT select the entire leaf.
    - Draw the box TIGHTLY around the most visible cluster of disease spots/lesions.
    - If there are multiple spots, box the largest group of spots.
    - Coordinates must be normalized (0-1000 scale).
    """
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                tools=[google_search_tool], 
                temperature=0.3,
                safety_settings=safety_settings,
            )
        )
        
        if not response.text:
            return {"error": "Agent returned Empty Response. Try a different photo."}

        search_sources = []
        if response.candidates[0].grounding_metadata.search_entry_point:
            search_sources.append(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)

        try:
            clean_text = clean_json_text(response.text)
            data = json.loads(clean_text)
            data['grounding_html'] = search_sources
            return data
        except json.JSONDecodeError:
            return {"error": f"Agent Error (Not JSON): {response.text[:100]}..."}

    except Exception as e:
        return {"error": f"Agent Failed: {str(e)}"}
    

st.title("🌱 AgriVision Pro")
st.caption("Autonomous Agent • Weather Aware • Market Search")

with st.sidebar:
    st.header("Settings")
    user_city = st.text_input("Your City", value="Sahiwal")

if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None

if st.session_state.captured_image is None:
    
    tab1, tab2 = st.tabs(["Camera", "Upload Image"])
    
    with tab1:
        img_file_buffer = st.camera_input("Scan your crop")
        if img_file_buffer:
            file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.session_state.captured_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.rerun()

    with tab2:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.session_state.captured_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.rerun()

else:
    image_spot = st.empty()
    image_spot.image(st.session_state.captured_image, caption="Agent is analyzing...", width="stretch")
    
    if st.button("New Scan"):
        st.session_state.captured_image = None
        st.rerun()

    with st.spinner(f"Agent is checking {user_city} weather & searching markets..."):
        pil_image = Image.fromarray(st.session_state.captured_image)
        result = plant_analyzer(pil_image, user_city)
        
        if "coordinates" in result:
            annotated_img = st.session_state.captured_image.copy()
            h, w, _ = annotated_img.shape
            
            coords = result.get('coordinates', [0, 0, 0, 0])
            ymin, xmin, ymax, xmax = coords
            
            start = (int(xmin * w / 1000), int(ymin * h / 1000))
            end = (int(xmax * w / 1000), int(ymax * h / 1000))
            
            cv2.rectangle(annotated_img, start, end, (57, 255, 20), 4)
            image_spot.image(annotated_img, caption=f"Detected: {result.get('disease_name')}", width="stretch")
            
            st.success(f"**Diagnosis:** {result.get('disease_name')}")
            st.warning(f"**Treatment Plan:** {result.get('treatment')}")
            st.info(f"**Recommended Medicine:** {result.get('medicine')}")
            
            st.markdown("---")
            st.write("**Agent's Market Research:**")
            st.write(result.get('search_finding', 'No external data needed.'))
            
            if 'grounding_html' in result and result['grounding_html']:
                st.markdown(result['grounding_html'][0], unsafe_allow_html=True)
            
        elif "error" in result:
            st.error(result['error'])
