import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from weather import get_weather

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

google_search_tool = types.Tool(google_search=types.GoogleSearch())


def analyze_plant_with_agent(image, city="Lahore"):
    
    try:
        prompt = f"""
        Diagnose disease in this image. User is in {city}, Pakistan.
        STEP 1: Identify the disease.
        STEP 2: Use the 'get_weather' tool to check if spraying is safe.
        STEP 3: Use Google Search to find the best medicine available in Pakistan for this disease.
        STEP 4: Return JSON with:
    - 'disease_name': str
    - 'treatment': str (include weather warning)
    - 'medicine': str (name of the product found)
    - 'coordinates': [ymin, xmin, ymax, xmax]
    - 'buy_link': str (The URL of the store or product page you found)
    """
       
        
        response = client.models.generate_content(
        # model="gemini-3-flash-preview", 
        model="gemini-3-flash-preview", 
        contents=[image, prompt],
        config=types.GenerateContentConfig(
        tools=[get_weather, google_search_tool], 
        system_instruction="You are an agent. Use Google Search to find real medicine links.",
        response_mime_type="application/json"
    )
)
        return json.loads(response.text)

    except Exception as e:
        return {
            "error": str(e),
            "disease_name": "Error",
            "coordinates": [0,0,0,0],
            "treatment": "System Error",
            "medicine": "N/A"
        }