import os
import requests
from dotenv import load_dotenv

# Load env vars if this file is run directly
load_dotenv()

def get_weather(city: str):
    """
    Fetches real-time weather data for a specific city. 
    Returns a string summary of temperature, condition, and humidity.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: Weather API Key is missing."

    # Free tier URL
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url).json()
        
        # Check if city was found
        if response.get("cod") != 200:
            return f"Weather unavailable for {city}."
            
        temp = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        condition = response["weather"][0]["description"]
        
        return f"Current Weather in {city}: {temp}°C, {condition}, Humidity: {humidity}%."
        
    except Exception as e:
        return f"Weather Tool Connection Error: {e}"

# Simple test block
if __name__ == "__main__":
    print(get_weather("Sahiwal"))