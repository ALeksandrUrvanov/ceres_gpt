import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('OPENAI_API_KEY')  # для OpenAI
bot_token = os.getenv('BOT_TOKEN')  # для Telegram бота