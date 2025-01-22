import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Optional
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram.enums import ChatAction
from main import VineyardAssistant
from config import bot_token

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VineyardBot:
    def __init__(self):
        self.bot = Bot(token=bot_token)
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.assistant: Optional[VineyardAssistant] = None
        self.cache = {}
        self.setup_handlers()
        self.is_running = True

    def setup_signal_handlers(self):
        """Настройка обработчиков сигналов"""
        signals = (signal.SIGINT, signal.SIGTERM)
        for sig in signals:
            signal.signal(sig, self.handle_shutdown)
            logger.info(f"Registered handler for signal {sig}")

    def handle_shutdown(self, signum, _):
        """Обработчик сигналов завершения"""
        logger.info(f"Received signal {signum}. Starting shutdown process...")
        self.is_running = False
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Корректное завершение работы бота"""
        try:
            logger.info("Shutting down bot...")
            await self.bot.session.close()
            await self.dp.storage.close()
            logger.info("Bot shutdown completed")
            sys.exit(0)
        except Exception as shutdown_error:
            logger.error(f"Error during shutdown: {shutdown_error}")
            sys.exit(1)

    async def initialize_assistant(self):
        """Ленивая инициализация ассистента"""
        if self.assistant is None:
            self.assistant = VineyardAssistant()
        return self.assistant

    def setup_handlers(self):
        """Настройка обработчиков сообщений"""
        self.dp.message.register(self.cmd_start, Command('start'))
        self.dp.message.register(self.cmd_help, Command('help'))
        self.dp.message.register(self.cmd_exit, Command('exit'))
        self.dp.message.register(self.handle_message)

    @staticmethod
    async def send_long_message(message: types.Message, text: str):
        """Отправка длинных сообщений по частям"""
        max_length = 4000
        parts = []

        while text:
            if len(text) <= max_length:
                parts.append(text)
                break

            part = text[:max_length]
            last_period = part.rfind('.')
            if last_period != -1:
                part = text[:last_period + 1]
                text = text[last_period + 1:].lstrip()
            else:
                text = text[max_length:]
            parts.append(part)

        for part in parts:
            await message.answer(part.strip())
            await asyncio.sleep(0.3)

    @staticmethod
    async def cmd_start(message: types.Message):
        """Обработчик команды /start"""
        welcome_text = (
            "Здравствуйте! Я ваш помощник-агроном по виноградарству компании Ceres Pro.\n"
            "Задайте мне вопрос и я постараюсь Вам помочь.\n\n"
            "Чтобы завершить диалог, используйте команду /exit\n"
            "Для получения помощи используйте команду /help"
        )
        await message.answer(welcome_text)
        logger.info(f"User {message.from_user.id} started conversation")

    @staticmethod
    async def cmd_help(message: types.Message):
        """Обработчик команды /help"""
        help_text = (
            "🍇 Я могу ответить на ваши вопросы о:\n"
            "- выращивании винограда\n"
            "- уходе за виноградной лозой\n"
            "- сортах винограда\n"
            "- обработке от болезней и вредителей\n"
            "- и многом другом\n\n"
            "Просто задайте свой вопрос в чат!\n"
            "Для завершения диалога используйте /exit"
        )
        await message.answer(help_text)

    @staticmethod
    async def cmd_exit(message: types.Message):
        """Обработчик команды /exit"""
        farewell_text = (
            "Спасибо за использование помощника-агронома. Удачи в виноградарстве!\n\n"
            "Чтобы начать новый диалог, используйте /start"
        )
        await message.answer(farewell_text)
        logger.info(f"User {message.from_user.id} ended conversation")

    async def get_cached_response(self, query: str) -> Optional[str]:
        """Получение ответа из кэша"""
        if query in self.cache:
            timestamp, response = self.cache[query]
            if (datetime.now() - timestamp).total_seconds() < 3600:
                return response
        return None

    async def cache_response(self, query: str, response: str):
        """Сохранение ответа в кэш"""
        self.cache[query] = (datetime.now(), response)

    async def handle_message(self, message: types.Message):
        # Логируем начало обработки запроса
        logger.info(f"Processing query: {message.text}")

        start_time = time.time()
        try:
            # Проверка кэша
            cached_response = await self.get_cached_response(message.text)
            if cached_response:
                await self.send_long_message(message, cached_response)
                await message.answer("\nЧтобы завершить диалог, используйте команду /exit")

                # Логирование времени обработки
                processing_time = time.time() - start_time
                logger.info(f"Cached query processed in {processing_time:.2f} seconds for user {message.from_user.id}")
                return

            await self.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

            # Инициализация ассистента при необходимости
            assistant = await self.initialize_assistant()

            # Обработка запроса
            response = await assistant.process_query(message.text)

            # Кэширование ответа
            await self.cache_response(message.text, response)

            # Отправка ответа
            await self.send_long_message(message, response)

            # Напоминание о команде выхода
            await message.answer("\nЧтобы завершить диалог, используйте команду /exit")

            # Логирование времени обработки
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds for user {message.from_user.id}")

        except Exception as message_error:
            error_message = (
                "Извините, произошла ошибка при обработке вашего запроса.\n"
                "Пожалуйста, попробуйте переформулировать вопрос или обратитесь позже."
            )
            await message.answer(error_message)
            logger.error(f"Error processing message from user {message.from_user.id}: {str(message_error)}", exc_info=True)

    async def start(self):
        """Запуск бота"""
        try:
            self.setup_signal_handlers()
            logger.info('Bot started')
            await self.dp.start_polling(self.bot)  # Используйте await
        except Exception as err:
            logger.error(f"Error starting bot: {str(err)}", exc_info=True)
        finally:
            await self.shutdown()

# Основной блок запуска
if __name__ == '__main__':
    vineyard_bot = VineyardBot()
    try:
        asyncio.run(vineyard_bot.start())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")