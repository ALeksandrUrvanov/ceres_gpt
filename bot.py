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
from aiogram.types import BotCommand, InlineKeyboardMarkup, InlineKeyboardButton
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
        self.is_running = True
        self.setup_handlers()

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

    async def setup_commands(self):
        """Настройка команд бота"""
        commands = [
            BotCommand(command="start", description="Начать диалог"),
            BotCommand(command="menu", description="Открыть дополнительное меню"),
            BotCommand(command="clear", description="Очистить историю диалога"),
            BotCommand(command="exit", description="Завершить диалог")
        ]
        await self.bot.set_my_commands(commands)

    async def initialize_assistant(self):
        """Ленивая инициализация ассистента"""
        if self.assistant is None:
            self.assistant = VineyardAssistant()
        return self.assistant

    def setup_handlers(self):
        """Настройка обработчиков сообщений"""
        self.dp.message.register(self.cmd_start, Command('start'))
        self.dp.message.register(self.cmd_menu, Command('menu'))
        self.dp.message.register(self.cmd_clear, Command('clear'))
        self.dp.message.register(self.cmd_exit, Command('exit'))
        self.dp.message.register(self.handle_message)
        self.dp.callback_query.register(self.handle_callback)

    @staticmethod
    def get_main_keyboard() -> InlineKeyboardMarkup:
        """Создание всплывающего меню с дополнительными функциями"""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🏢 О компании", callback_data="about_company"),
                InlineKeyboardButton(text="🧹 Очистить историю", callback_data="clear_history")
            ],
            [
                InlineKeyboardButton(text="ℹ️ О боте", callback_data="about"),
                InlineKeyboardButton(text="❌ Завершить диалог", callback_data="exit")
            ]
        ])
        return keyboard

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

    async def handle_callback(self, callback_query: types.CallbackQuery):
        """Обработка нажатий на кнопки во всплывающем меню"""
        data = callback_query.data
        if data == "about_company":
            await self.cmd_about_company(callback_query.message)
        elif data == "clear_history":
            await self.cmd_clear(callback_query.message)
        elif data == "about":
            await self.show_about(callback_query.message)
        elif data == "exit":
            await self.cmd_exit(callback_query.message)

        await callback_query.answer()

    @staticmethod
    async def show_about(message: types.Message):
        """Показ информации о боте (О боте)"""
        about_text = (
            "Vineyard Assistant Bot (VAB)\n\n"
            "Специализированный бот-помощник по виноградарству.\n"
            "Использует современные технологии AI для предоставления точной и полезной информации.\n\n"
            "Version: 2.0\n"
            "Created by: Ceres Pro"
        )
        await message.answer(about_text)

    @staticmethod
    async def cmd_about_company(message: types.Message):
        """Показ информации о компании (ранее команда /help)"""
        about_company_text = (
            "Компания Ceres Pro | Метеосистемы для агрохозяйств\n"
            "Ceres Pro — это не стандартная метеостанция, а индивидуально подобранное решение для вашего виноградника.\n\n"
            "Техподдержка, сервис, гарантия:\n"
            "г. Севастополь, ул. Хрусталёва, 74А\n\n"
            "+7 (978) 858-55-25\n"
            "info@proceres.ru\n"
            "@pro_ceres\n"
            "proceres.ru"
        )
        await message.answer(about_company_text)

    @staticmethod
    async def cmd_start(message: types.Message):
        """Обработчик команды /start (приветственное сообщение без меню)"""
        welcome_text = (
            "Здравствуйте! Я ваш помощник-агроном по виноградарству компании Ceres Pro.\n"
            "Задайте мне вопрос, и я постараюсь вам помочь.\n\n"
            "Для доступа к дополнительным функциям используйте команду /menu.\n"
            "Для завершения диалога используйте команду /exit."
        )
        await message.answer(welcome_text)
        logger.info(f"User {message.from_user.id} started conversation")

    async def cmd_menu(self, message: types.Message):
        """Обработчик команды /menu для показа всплывающего меню"""
        await message.answer("Доступные опции:", reply_markup=self.get_main_keyboard())

    async def cmd_clear(self, message: types.Message):
        """Сбрасывает историю диалога"""
        if self.assistant and hasattr(self.assistant, 'session_manager'):
            self.assistant.session_manager.sessions.pop(message.from_user.id, None)
        await message.answer("История диалога сброшена. Вы можете начать новый разговор.")

    async def cmd_exit(self, message: types.Message):
        """Обработчик команды /exit и кнопки 'Завершить диалог'"""
        if self.assistant:
            self.assistant.session_manager.sessions.pop(message.from_user.id, None)
        farewell_text = (
            "Спасибо за использование помощника-агронома. Удачи в виноградарстве!\n\n"
            "Чтобы начать новый диалог, используйте команду /start."
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
        """Обработка входящих сообщений"""
        logger.info(f"Processing query: {message.text}")
        start_time = time.time()

        try:
            # Проверка кэша
            cached_response = await self.get_cached_response(message.text)
            if cached_response:
                await self.send_long_message(message, cached_response)
                processing_time = time.time() - start_time
                logger.info(
                    f"Cached query processed in {processing_time:.2f} seconds for user {message.from_user.id}"
                )
                return

            # Уведомляем пользователя, что бот обрабатывает запрос
            await self.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

            # Инициализация ассистента при необходимости
            assistant = await self.initialize_assistant()

            try:
                response = await assistant.process_query(message.text, message.from_user.id)
            except Exception as e:
                error_str = str(e)
                if "Rate limit reached" in error_str:
                    # Извлекаем время ожидания из сообщения об ошибке
                    wait_time = "несколько минут"
                    if "try again in" in error_str:
                        time_str = error_str.split("try again in")[1].split('.')[0].strip()
                        wait_time = time_str

                    error_message = (
                        "Извините, достигнут лимит запросов к API.\n"
                        f"Пожалуйста, подождите {wait_time} перед следующим запросом."
                    )
                    await message.answer(error_message)
                    logger.error(f"Rate limit exceeded for user {message.from_user.id}: {error_str}")
                    return
                elif "token limit" in error_str.lower():
                    await message.answer(
                        "Ваш запрос слишком длинный и превышает допустимый лимит токенов. "
                        "Пожалуйста, сократите текст и попробуйте снова."
                    )
                    logger.error(f"Token limit exceeded for user {message.from_user.id}: {error_str}")
                    return
                else:
                    raise

            # Кэширование ответа
            await self.cache_response(message.text, response)

            # Отправка ответа пользователю
            await self.send_long_message(message, response)

            # Логирование времени обработки
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds for user {message.from_user.id}")

        except Exception as message_error:
            error_str = str(message_error)
            if "Rate limit reached" in error_str:
                # Извлекаем время ожидания из сообщения об ошибке
                wait_time = "несколько минут"
                if "try again in" in error_str:
                    time_str = error_str.split("try again in")[1].split('.')[0].strip()
                    wait_time = time_str

                error_message = (
                    "Извините, достигнут лимит запросов к API.\n"
                    f"Пожалуйста, подождите {wait_time} перед следующим запросом.\n"
                    "Это ограничение установлено для обеспечения стабильной работы сервиса."
                )
            else:
                error_message = (
                    "Извините, произошла ошибка при обработке вашего запроса.\n"
                    "Пожалуйста, попробуйте позже или переформулируйте ваш вопрос."
                )
            await message.answer(error_message)
            logger.error(f"Error processing message from user {message.from_user.id}: {error_str}", exc_info=True)

    async def start(self):
        """Запуск бота"""
        try:
            self.setup_signal_handlers()
            await self.setup_commands()
            logger.info('Bot started')
            await self.dp.start_polling(self.bot)
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
    except Exception as unexpected_error:
        logger.error(f"Unexpected error: {unexpected_error}")