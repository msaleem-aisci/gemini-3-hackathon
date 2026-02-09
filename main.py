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
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API Key missing.")
    st.stop()

client = genai.Client(api_key=api_key)


google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)

def get_weather_context(city):
    """Fetches real-time weather to inject into the Agent's brain."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key: return "Weather data unavailable."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
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
    2. REASON about the Weather: Look at the weather data above. If humidity is high or rain is present, warn the user about spraying.
    3. SEARCH GOOGLE: Use your search tool to find the price/availability of the best medicine in Pakistan.
    
    OUTPUT:
    Return strictly valid JSON:
    {{
        "disease_name": "str",
        "treatment": "str (Combine your medical knowledge with the weather warning)",
        "medicine": "str",
        "coordinates": [ymin, xmin, ymax, xmax],
        "search_finding": "str (Summary of the online prices you found)"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                tools=[google_search_tool], 
                temperature=0.4
            )
        )
        
        search_sources = []
        if response.candidates[0].grounding_metadata.search_entry_point:
            search_sources.append(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)

        clean_text = clean_json_text(response.text)
        data = json.loads(clean_text)
        data['grounding_html'] = search_sources
        
        return data

    except Exception as e:
        return {"error": f"Agent Failed: {str(e)}"}

st.title("AgriVision Pro: Autonomous Agent")
st.caption("Powered by Gemini • Weather Aware • Live Search")

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
    image_spot.image(st.session_state.captured_image, caption="Agent is analyzing...", width="stretch")
    
    if st.button("🔄 New Scan"):
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
            
            # RESULTS
            st.success(f"**Diagnosis:** {result.get('disease_name')}")
            st.warning(f"**Treatment Plan:** {result.get('treatment')}")
            st.info(f"**Recommended Medicine:** {result.get('medicine')}")
            
            # SHOW THE AGENT'S "THOUGHTS"
            st.markdown("---")
            st.write("*Agent's Market Research:**")
            st.write(result.get('search_finding', 'No external data needed.'))
            
            # GOOGLE SEARCH WIDGET (The Proof)
            if 'grounding_html' in result and result['grounding_html']:
                st.markdown(result['grounding_html'][0], unsafe_allow_html=True)
            
        elif "error" in result:
            st.error(result['error'])
