import requests

"""
Weather in London:
- Conditions: Light Rain
- Temperature: 15.7°C
- Feels like: 15.2°C
- Humidity: 85%
- Wind: 3.6 m/s
"""


def validate_api_key(api_key: str) -> bool:
    """
    Validate an OpenWeatherMap API key by making a test request.

    Args:
        api_key (str): The API key to validate

    Returns:
        bool: True if the API key is valid, False otherwise
    """
    if not api_key or not api_key.strip():
        return False

    # Use a simple test query (London is a common city that should work)
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": "London,UK", "units": "metric", "appid": api_key}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        # Check if the response indicates an invalid API key
        if data.get("cod") == 401:
            # 401 Unauthorized typically means invalid API key
            return False
        elif data.get("cod") == 200:
            # 200 OK means the API key works
            return True
        else:
            # Other errors might be temporary, but we'll consider key invalid
            return False

    except requests.exceptions.RequestException:
        # Network errors or other issues - assume invalid for safety
        return False
    except Exception:
        # Any other exception - assume invalid
        return False


def format_weather_message(weather_data: dict) -> str:
    """
    Format weather data into a readable message string.

    Args:
        weather_data (dict): Weather data dictionary from get_current_weather()

    Returns:
        str: Formatted weather message
    """
    return (
        f"Weather in {weather_data['city']}:\n"
        f"- Conditions: {weather_data['description']}\n"
        f"- Temperature: {weather_data['temperature']}{weather_data['unit']}\n"
        f"- Feels like: {weather_data['feels_like']}{weather_data['unit']}\n"
        f"- Humidity: {weather_data['humidity']}%\n"
        f"- Wind: {weather_data['wind_speed']} {weather_data['wind_unit']}"
    )


def get_current_weather(api_key: str, city: str, units: str = "metric") -> dict:
    """
    Get current weather data for a city

    Args:
        api_key (str): Your OpenWeatherMap API key (must be valid)
        city (str): City name (e.g., "London,UK")
        units (str): 'metric' (Celsius) or 'imperial' (Fahrenheit)

    Returns:
        dict: Weather data dictionary with keys: city, description, temperature,
              feels_like, humidity, wind_speed, unit, wind_unit

    Raises:
        ValueError: If the API key is invalid or the request fails
        requests.exceptions.RequestException: If there's a network error
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is required and cannot be empty")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "units": units, "appid": api_key}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()

        if data["cod"] != 200:
            error_msg = data.get("message", "Unknown error")
            if data["cod"] == 401:
                raise ValueError(f"Invalid API key: {error_msg}")
            else:
                raise ValueError(f"API Error {data['cod']}: {error_msg}")

        return {
            "city": data["name"],
            "description": data["weather"][0]["description"].title(),
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "unit": "°C" if units == "metric" else "°F",
            "wind_unit": "m/s" if units == "metric" else "mph",
        }

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Weather request failed: {e}")


# Example usage
if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
    CITY = "London,UK"

    try:
        # Validate API key first
        if not validate_api_key(API_KEY):
            print("Error: Invalid API key")
            exit(1)

        weather = get_current_weather(API_KEY, CITY)
        print(format_weather_message(weather))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
