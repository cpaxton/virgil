import requests

"""
Weather in London:
- Conditions: Light Rain
- Temperature: 15.7°C
- Feels like: 15.2°C
- Humidity: 85%
- Wind: 3.6 m/s
"""

def get_current_weather(api_key, city, units='metric'):
    """
    Get current weather data for a city
    
    Args:
        api_key (str): Your OpenWeatherMap API key
        city (str): City name (e.g., "London,UK")
        units (str): 'metric' (Celsius) or 'imperial' (Fahrenheit)
    
    Returns:
        dict: Weather data dictionary
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'units': units,
        'appid': api_key
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        
        if data['cod'] != 200:
            raise ValueError(f"API Error {data['cod']}: {data.get('message', 'Unknown error')}")
        
        return {
            'city': data['name'],
            'description': data['weather'][0]['description'].title(),
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'unit': '°C' if units == 'metric' else '°F',
            'wind_unit': 'm/s' if units == 'metric' else 'mph'
        }
        
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Weather request failed: {e}")

# Example usage
if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
    CITY = "London,UK"
    
    try:
        weather = get_current_weather(API_KEY, CITY)
        print(f"Weather in {weather['city']}:")
        print(f"- Conditions: {weather['description']}")
        print(f"- Temperature: {weather['temperature']}{weather['unit']}")
        print(f"- Feels like: {weather['feels_like']}{weather['unit']}")
        print(f"- Humidity: {weather['humidity']}%")
        print(f"- Wind: {weather['wind_speed']} {weather['wind_unit']}")
    except Exception as e:
        print(f"Error: {e}")


    
