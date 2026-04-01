import os
import warnings

# Suppress Pydantic V1 compatibility warning from langchain_core
warnings.filterwarnings("ignore", message=".*Core Pydantic V1 functionality.*")

from dotenv import load_dotenv

from src.bot import BetterChatbot
from src.settings import get_logger

logger = get_logger(__name__)
load_dotenv()

# Load course material
with open("assets/course_notes.txt", "r", encoding="utf-8") as f:
    course_text = f.read()

bot = BetterChatbot(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
    course_text=course_text,
)

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        logger.info("👋 Conversation ended by user.")
        break
    response = bot.chat(user_input)
    print("Bot:", response)


