import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram.enums import ChatAction
from main import VineyardAssistant
from config import bot_token



# Инициализация бота и диспетчера
bot = Bot(token=bot_token)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Инициализация ассистента
assistant = VineyardAssistant()


@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    welcome_text = (
        "Здравствуйте! Я ваш помощник-агроном по виноградарству.\n\n"
        "Задайте мне вопрос о выращивании винограда, и я постараюсь помочь.\n"
        "Чтобы завершить диалог, используйте команду /exit\n"
        "Для получения помощи используйте команду /help"
    )
    await message.answer(welcome_text)


@dp.message(Command('help'))
async def cmd_help(message: types.Message):
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


@dp.message(Command('exit'))
async def cmd_exit(message: types.Message):
    farewell_text = "Спасибо за использование помощника-агронома. Удачи в виноградарстве!\n\nЧтобы начать новый диалог, используйте /start"
    await message.answer(farewell_text)


@dp.message()
async def handle_message(message: types.Message):
    try:
        # Если сообщение похоже на команду выхода
        if message.text.lower() in ['выход', 'exit', 'quit']:
            await cmd_exit(message)
            return

        # Показываем пользователю, что бот печатает
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

        # Обработка запроса в отдельном потоке
        response = await asyncio.to_thread(
            assistant.process_query,
            message.text
        )

        # Если ответ слишком длинный, разбиваем его на части
        if len(response) > 4000:
            parts = [response[i:i + 4000] for i in range(0, len(response), 4000)]
            for part in parts:
                await message.answer(part)
        else:
            await message.answer(response)

        # Предложение задать следующий вопрос
        await message.answer(
            "\nОстались ли ещё вопросы? Задавайте! "
            "Если хотите завершить диалог, используйте команду /exit"
        )

    except Exception as e:
        error_message = (
            "Извините, произошла ошибка при обработке вашего запроса.\n"
            "Пожалуйста, попробуйте переформулировать вопрос или обратитесь позже."
        )
        await message.answer(error_message)
        logging.error(f"Error in message handler: {e}", exc_info=True)


async def main():
    # Включаем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info('Бот запущен')

    # Запускаем бота
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info('Бот остановлен')