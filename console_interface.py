import logging
import asyncio
import time
from main import VineyardAssistant

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vineyard_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConsoleInterface:
    def __init__(self):
        """Инициализация консольного интерфейса"""
        logger.info("Initializing Vineyard Assistant...")
        try:
            self.assistant = VineyardAssistant()
            self.user_id = 1  # Фиксированный ID для консольной версии
            logger.info("Vineyard Assistant initialized successfully")
        except Exception as failed_to_initialize_vineyard_assistant:
            logger.error(f"Failed to initialize Vineyard Assistant: {failed_to_initialize_vineyard_assistant}", exc_info=True)
            raise

    async def process_query_and_show_details(self, query: str):
        """Обработка запроса с выводом подробной информации"""
        start_time = time.time()

        try:
            # Предобработка запроса
            processed_query = self.assistant.preprocess_query(query)
            print(f"\nПредобработанный запрос: {processed_query}")

            # Поиск похожих документов
            similar_docs = self.assistant.get_similar_documents(processed_query, k=5)

            # Вывод найденных документов
            print("\n=== Найденные похожие документы ===")
            for i, doc in enumerate(similar_docs, 1):
                print(f"\nДокумент {i}:")
                print("-" * 50)
                print(doc.page_content)
                print("-" * 50)

            # Получение ответа с учетом контекста
            response = await self.assistant.process_query(query, self.user_id)

            # Вычисляем время обработки
            processing_time = time.time() - start_time

            # Логируем время обработки
            logger.info(f"Processing query: {query}")
            logger.info(f"Query processed in {processing_time:.2f} seconds")

            return response

        except Exception as error_in_query_processing:
            logger.error(f"Error in query processing: {error_in_query_processing}", exc_info=True)
            return "Произошла ошибка при обработке запроса."

    @staticmethod
    def show_commands():
        """Вывод списка доступных команд"""
        print("\n=== Доступные команды ===")
        print("выход - завершить работу")
        print("clear - очистить историю диалога")
        print("help  - показать это сообщение")
        print("context - показать текущий контекст диалога")
        print("========================\n")

    def show_context(self):
        """Вывод текущего контекста диалога"""
        if self.assistant and self.user_id in self.assistant.session_manager.sessions:
            context = self.assistant.session_manager.get_context(self.user_id)
            print("\n=== Текущий контекст диалога ===")
            for msg in context:
                role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                print(f"\n{role}:")
                print("-" * 50)
                print(msg["content"])
                print("-" * 50)
        else:
            print("\nИстория диалога пуста")

    async def run(self):
        """Запуск консольного интерфейса"""
        try:
            print("Здравствуйте! Я ваш помощник-агроном по виноградарству компании Ceres Pro.")
            print("Задайте мне вопрос и я постараюсь Вам помочь.")
            self.show_commands()

            while True:
                try:
                    query = input("\nВаш вопрос: ").strip().lower()

                    if not query:
                        print("Пожалуйста, введите ваш вопрос.")
                        continue

                    # Обработка команд
                    if query == 'help':
                        self.show_commands()
                        continue
                    elif query == 'context':
                        self.show_context()
                        continue
                    elif query == 'clear':
                        if self.assistant:
                            self.assistant.session_manager.sessions.pop(self.user_id, None)
                            print("\nИстория диалога очищена")
                        continue
                    elif self._is_exit_command(query):
                        print("\nСпасибо за использование помощника Ceres Pro!")
                        print("До свидания и удачи в виноградарстве!\n")
                        break

                    # Обработка запроса с выводом информации
                    response = await self.process_query_and_show_details(query)

                    # Вывод финального ответа
                    print("\n=== Ответ помощника ===")
                    print(response)

                except KeyboardInterrupt:
                    print("\n\nРабота программы прервана пользователем.")
                    break
                except Exception as error_processing_query:
                    logger.error(f"Error processing query: {error_processing_query}", exc_info=True)
                    print("\nПроизошла ошибка при обработке запроса.")
                    print("Пожалуйста, попробуйте еще раз или переформулируйте вопрос.")

        except Exception as unexpected_error_in_console:
            logger.error(f"Unexpected error in console interface: {unexpected_error_in_console}", exc_info=True)
            print("\nПроизошла непредвиденная ошибка. Программа будет завершена.")
        finally:
            # Очистка сессии при завершении
            if self.assistant:
                self.assistant.session_manager.sessions.pop(self.user_id, None)
            logger.info("Console interface shutdown")

    @staticmethod
    def _is_exit_command(query: str) -> bool:
        """Проверка команды выхода"""
        return query in ['выход', 'exit', 'quit']


async def amain():
    """Асинхронная точка входа в программу"""
    try:
        console = ConsoleInterface()
        await console.run()
    except Exception as failed_to_start_console:
        logger.error(f"Failed to start console interface: {failed_to_start_console}", exc_info=True)
        print("\nНе удалось запустить программу из-за критической ошибки.")
        print("Проверьте лог-файл для получения дополнительной информации.")

def main():
    """Точка входа в программу"""
    asyncio.run(amain())

if __name__ == "__main__":
    main()