import os
import requests

from dotenv import load_dotenv
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

api_key = os.getenv("OPENWEATHER_API_KEY")


@tool("Weather Tool")
def get_weather(city: str) -> str:
    """Get the current weather information for a city."""

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Could not get weather for {city}. Error: {response.json()}"

    data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data["wind"]["speed"]

    return (
        f"City: {city}\n"
        f"Temperature: {temperature} °C\n"
        f"Humidity: {humidity}%\n"
        f"Weather: {description}\n"
        f"Wind Speed: {wind_speed} m/s"
    )


weather_agent = Agent(
    role="Weather Assistant",
    goal="Get accurate current weather information for a city",
    backstory="You are a helpful weather assistant who uses the Weather Tool to provide current weather information.",
    tools=[get_weather],
    llm=llm,
    verbose=True
)

city = input("Enter the city name: ")

weather_task = Task(
    description=f"Get the current weather information for {city} using the Weather Tool.",
    expected_output="A clear report containing the city, temperature, humidity, weather description, and wind speed.",
    agent=weather_agent
)

weather_crew = Crew(
    agents=[weather_agent],
    tasks=[weather_task],
    process=Process.sequential,
    verbose=True
)

result = weather_crew.kickoff()

print("\nFINAL RESULT:")
print(result)