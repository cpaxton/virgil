# Weather API Setup Guide

The Friend Discord bot can provide weather information to users when configured with an OpenWeatherMap API key. This guide will walk you through obtaining and configuring the API key.

## What is OpenWeatherMap?

OpenWeatherMap is a weather data service that provides current weather, forecasts, and historical weather data through a RESTful API. The Friend bot uses their free tier to fetch current weather conditions for any city worldwide.

## Getting an API Key

### Step 1: Create an Account

1. Go to [OpenWeatherMap.org](https://openweathermap.org/)
2. Click **"Sign Up"** in the top right corner
3. Fill out the registration form:
   - Email address
   - Username
   - Password
   - Confirm you agree to the Terms of Service

### Step 2: Verify Your Email

1. Check your email inbox for a verification message from OpenWeatherMap
2. Click the verification link in the email
3. You'll be redirected to the OpenWeatherMap website

### Step 3: Get Your API Key

1. Once logged in, navigate to the **API keys** section:
   - Click on your username in the top right
   - Select **"My API keys"** from the dropdown menu
   - Or go directly to: https://home.openweathermap.org/api_keys

2. You'll see a default API key (or you can create a new one):
   - The default key is named "Default"
   - You can create multiple keys for different projects
   - Click **"Generate"** if you want to create a new key

3. **Copy your API key** - you'll need this to configure the bot

### Step 4: Understand API Limits

The free tier includes:
- **60 calls/minute** - More than enough for a Discord bot
- **1,000,000 calls/month** - Generous limit for personal use
- Current weather data only
- No credit card required

**Note**: The API key may take a few minutes to activate after creation. If you get validation errors immediately after creating it, wait 5-10 minutes and try again.

## Configuring the Bot

There are two ways to provide the API key to the Friend bot:

### Method 1: Environment Variable (Recommended)

This is the most secure method, especially for production deployments:

1. **Set the environment variable**:

   On Linux/macOS:
   ```bash
   export OPENWEATHER_API_KEY=your_api_key_here
   ```

   On Windows (PowerShell):
   ```powershell
   $env:OPENWEATHER_API_KEY="your_api_key_here"
   ```

   On Windows (Command Prompt):
   ```cmd
   set OPENWEATHER_API_KEY=your_api_key_here
   ```

2. **Add to your shell configuration** (optional, for persistence):

   Add to `~/.bashrc` or `~/.zshrc`:
   ```bash
   export OPENWEATHER_API_KEY=your_api_key_here
   ```

   Then reload:
   ```bash
   source ~/.bashrc
   ```

3. **Run the bot**:
   ```bash
   python -m virgil.friend.friend --token $DISCORD_TOKEN
   ```

   The bot will automatically detect the API key from the environment variable.

### Method 2: Command Line Argument

You can pass the API key directly when starting the bot:

```bash
python -m virgil.friend.friend --token $DISCORD_TOKEN --weather-api-key your_api_key_here
```

**Note**: This method is less secure as the API key may appear in process lists or shell history.

## Verification

When you start the bot, it will automatically validate your API key:

- ✅ **Valid key**: You'll see `"Weather API key is valid."` in the console
- ❌ **Invalid key**: You'll see a warning: `"Warning: Weather API key appears to be invalid. Weather functionality will be disabled."`
- ⚠️ **No key**: You'll see `"No weather API key provided. Weather functionality will be disabled."`

If your key is invalid, check:
1. Did you copy the entire key correctly?
2. Has the key been activated? (New keys take 5-10 minutes)
3. Are there any extra spaces or characters?

## Using Weather in Discord

Once configured, users can ask the bot about weather:

**Example prompts:**
- "What's the weather in London?"
- "How's the weather in New York?"
- "Tell me the weather in Tokyo, Japan"

The bot will automatically:
1. Parse the city name from the user's message
2. Fetch current weather data
3. Display formatted weather information including:
   - Current conditions
   - Temperature (and feels like)
   - Humidity
   - Wind speed

**Example bot response:**
```
Weather in London:
- Conditions: Light Rain
- Temperature: 15.7°C
- Feels like: 15.2°C
- Humidity: 85%
- Wind: 3.6 m/s
```

## City Name Format

The bot accepts city names in various formats:
- **City only**: `London` (may be ambiguous)
- **City, Country**: `London,UK` (recommended)
- **City, State, Country**: `New York,NY,US`
- **City, Country Code**: `Paris,FR`

Using the country code helps avoid ambiguity when multiple cities share the same name.

## Troubleshooting

### Bot says "API key is not configured"
- Check that you've set the environment variable or passed `--weather-api-key`
- Verify the variable name is exactly `OPENWEATHER_API_KEY` (case-sensitive)

### Bot says "API key is invalid"
- Verify you copied the entire key (no missing characters)
- Check that the key is activated (wait 5-10 minutes after creation)
- Try regenerating the key in the OpenWeatherMap dashboard
- Verify the key hasn't been revoked or expired

### Weather query fails
- Check that the city name is spelled correctly
- Try including the country code (e.g., `London,UK` instead of just `London`)
- Verify your API key hasn't exceeded rate limits (60 calls/minute)
- Check your internet connection

### Rate Limiting
If you exceed 60 calls per minute, you'll receive rate limit errors. The bot will handle these gracefully and inform users. For most Discord servers, this limit is more than sufficient.

## Security Best Practices

1. **Never commit API keys to version control**
   - Add `OPENWEATHER_API_KEY` to your `.gitignore` if storing in a file
   - Use environment variables in production

2. **Rotate keys if compromised**
   - If you suspect your key is compromised, regenerate it in the OpenWeatherMap dashboard
   - Update the environment variable or configuration

3. **Monitor usage**
   - Check your OpenWeatherMap dashboard periodically to monitor API usage
   - Set up alerts if you're approaching limits

4. **Use separate keys for different environments**
   - Use one key for development/testing
   - Use a different key for production
   - This helps isolate issues and track usage

## Additional Resources

- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [OpenWeatherMap Pricing](https://openweathermap.org/price) (free tier is usually sufficient)
- [OpenWeatherMap Support](https://openweathermap.org/faq)

## Example Complete Setup

Here's a complete example of setting up the bot with weather support:

```bash
# 1. Set Discord token (from Discord Developer Portal)
export DISCORD_TOKEN=your_discord_bot_token

# 2. Set Weather API key (from OpenWeatherMap)
export OPENWEATHER_API_KEY=your_openweather_api_key

# 3. Run the bot
python -m virgil.friend.friend --backend gemma-2-2b-it

# The bot will:
# - Connect to Discord
# - Validate the weather API key
# - Be ready to answer weather queries
```

Now your bot is ready to provide weather information to users on Discord!

