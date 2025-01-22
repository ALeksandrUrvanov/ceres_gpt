import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('OPENAI_API_KEY')  # для OpenAI
bot_token = os.getenv('BOT_TOKEN')  # для Telegram бота

# Параметры для обработки текста
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Параметры моделей
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GPT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3
MAX_TOKENS = 1500

# Параметры векторного поиска
DEFAULT_SIMILAR_DOCS_COUNT = 5