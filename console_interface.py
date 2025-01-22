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
            logger.info("Vineyard Assistant initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vineyard Assistant: {e}", exc_info=True)
            raise

    async def process_query_and_show_details(self, query: str):
        """Обработка запроса с выводом подробной информации"""
        # Засекаем начальное время
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

            # Получение ответа
            response = await self.assistant.process_query(query)

            # Вычисляем время обработки
            processing_time = time.time() - start_time

            # Логируем время обработки
            logger.info(f"Processing query: {query}")
            logger.info(f"Query processed in {processing_time:.2f} seconds")

            return response

        except Exception as e:
            logger.error(f"Error in query processing: {e}", exc_info=True)
            return "Произошла ошибка при обработке запроса."

    async def run(self):
        """Запуск консольного интерфейса"""
        try:
            print("Здравствуйте! Я ваш помощник-агроном по виноградарству компании Ceres Pro.")
            print("Задайте мне вопрос и я постараюсь Вам помочь.")
            print("Для выхода введите 'выход'")

            while True:
                try:
                    # Получение запроса от пользователя
                    query = input("\nВаш вопрос: ").strip()

                    # Проверка команды выхода
                    if self._is_exit_command(query):
                        print("\nСпасибо за использование помощника Ceres Pro!")
                        print("До свидания и удачи в виноградарстве!\n")
                        break

                    if not query:
                        print("Пожалуйста, введите ваш вопрос.")
                        continue

                    # Обработка запроса с выводом информации
                    response = await self.process_query_and_show_details(query)

                    # Вывод финального ответа
                    print("\n=== Ответ помощника ===")
                    print(response)

                except KeyboardInterrupt:
                    print("\n\nРабота программы прервана пользователем.")
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {e}", exc_info=True)
                    print("\nПроизошла ошибка при обработке запроса.")
                    print("Пожалуйста, попробуйте еще раз или переформулируйте вопрос.")

        except Exception as e:
            logger.error(f"Unexpected error in console interface: {e}", exc_info=True)
            print("\nПроизошла непредвиденная ошибка. Программа будет завершена.")
        finally:
            logger.info("Console interface shutdown")

    @staticmethod
    def _is_exit_command(query: str) -> bool:
        """Проверка команды выхода"""
        return query.lower() in ['выход', 'exit', 'quit']


async def amain():
    """Асинхронная точка входа в программу"""
    try:
        console = ConsoleInterface()
        await console.run()
    except Exception as e:
        logger.error(f"Failed to start console interface: {e}", exc_info=True)
        print("\nНе удалось запустить программу из-за критической ошибки.")
        print("Проверьте лог-файл для получения дополнительной информации.")

def main():
    """Точка входа в программу"""
    asyncio.run(amain())

if __name__ == "__main__":
    main()