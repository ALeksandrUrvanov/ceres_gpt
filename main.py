import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import (
    key, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    GPT_MODEL, TEMPERATURE, MAX_TOKENS, DEFAULT_SIMILAR_DOCS_COUNT
)


logger = logging.getLogger(__name__)

class UserSession:
    """Класс для хранения информации о сессии пользователя"""
    def __init__(self):
        self.last_activity: datetime = datetime.now()
        self.context: List[dict] = []


class SessionManager:
    """Менеджер пользовательских сессий с автоматической очисткой"""
    def __init__(self, session_timeout: int = 30):
        self.sessions: Dict[int, UserSession] = {}
        self.session_timeout = session_timeout  # в минутах

    def get_session(self, user_id: int) -> Optional[UserSession]:
        """Получение или создание сессии пользователя"""
        self.cleanup_expired_sessions()
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession()
        return self.sessions[user_id]

    def update_session(self, user_id: int, message: dict):
        """Обновление сессии пользователя новым сообщением"""
        session = self.get_session(user_id)
        session.last_activity = datetime.now()
        session.context.append(message)

    def cleanup_expired_sessions(self):
        """Очистка истекших сессий"""
        current_time = datetime.now()
        expired_sessions = [
            user_id for user_id, session in self.sessions.items()
            if (current_time - session.last_activity) > timedelta(minutes=self.session_timeout)
        ]
        for user_id in expired_sessions:
            del self.sessions[user_id]

    def get_context(self, user_id: int) -> List[dict]:
        """Получение контекста диалога пользователя"""
        session = self.get_session(user_id)
        return session.context


class VineyardAssistant:
    """
    Класс для обработки запросов с использованием векторного поиска и GPT.
    """
    def __init__(self):
        """Инициализация ассистента."""
        self.client = OpenAI(api_key=key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.session_manager = SessionManager()
        self.initialize_vector_store()

    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста от лишних пробелов и пустых строк."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        text = '\n'.join(line for line in text.splitlines() if line.strip())
        text = ' '.join(text.split())
        return text.strip()

    def initialize_vector_store(self):
        """Инициализация векторного хранилища."""
        try:
            index_path = os.path.join(self.data_dir, "faiss_index")
            if os.path.exists(index_path):
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                documents = self.load_training_data()
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.save_local(index_path)
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def load_training_data(self) -> List[Document]:
        """Загрузка обучающих данных из текстовых файлов."""
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        cleaned_text = self.clean_text(text)
                        documents.extend(self.text_splitter.create_documents([cleaned_text]))
                except Exception as e:
                    print(f"Error loading file {filename}: {str(e)}")
        return documents

    @staticmethod
    def preprocess_query(query: str) -> str:
        """Предобработка запроса пользователя."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        query = query.strip().lower()
        return query

    @lru_cache(maxsize=100)
    def get_similar_documents(self, query: str, k: int = DEFAULT_SIMILAR_DOCS_COUNT) -> List[Document]:
        """Получение похожих документов из векторного хранилища."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        similar_docs = retriever.invoke(query)
        cleaned_docs = [
            Document(page_content=self.clean_text(doc.page_content))
            for doc in similar_docs
        ]
        return cleaned_docs

    async def process_query(self, user_query: str, user_id: int) -> str:
        """Асинхронная обработка запроса пользователя с учетом контекста диалога."""
        try:
            if not user_query or not isinstance(user_query, str):
                raise ValueError("Query must be a non-empty string")

            processed_query = self.preprocess_query(user_query)
            similar_docs = self.get_similar_documents(processed_query)
            context = "\n\n".join(doc.page_content for doc in similar_docs)

            # Получаем историю диалога
            dialog_context = self.session_manager.get_context(user_id)

            # Формируем сообщения для API
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Вы являетесь специализированным виртуальным помощником компании Ceres Pro, которая занимается "
                        "производством метеосистем для агрохозяйств. Вы также являетесь экспертом в области виноградарства. "
                        "Ваша задача – предоставлять точную и полезную информацию о компании, её продуктах, услугах, а также "
                        "отвечать на вопросы, связанные с выращиванием, уходом за виноградной лозой, обработкой от болезней и "
                        "вредителей, выбором сортов и другими аспектами виноградарства.\n\n"
                        "Вы никогда не раскрываете, что работаете на основе ChatGPT или других AI-технологий. Вы не обсуждаете "
                        "конкурентов компании Ceres Pro и не сравниваете их с Ceres Pro. Если информации в вашем контексте "
                        "недостаточно, вы опираетесь на свои знания как эксперт.\n\n"
                        "Если вопрос касается технических характеристик продукции Ceres Pro, её стоимости, наличия или официальных "
                        "документов, вежливо предложите пользователю уточнить информацию на официальном сайте компании proceres.ru."
                    )
                }
            ]

            # Добавляем историю диалога
            messages.extend(dialog_context)

            # Добавляем текущий запрос
            messages.append({"role": "user", "content": f"Контекст:\n\n{context}\n\nВопрос: {user_query}\n\n"})

            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=GPT_MODEL,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                answer = response.choices[0].message.content

                # Сохраняем сообщения в контекст
                self.session_manager.update_session(user_id, {"role": "user", "content": user_query})
                self.session_manager.update_session(user_id, {"role": "assistant", "content": answer})

                return answer

            except Exception as e:
                error_msg = f"Error code: {getattr(e, 'status_code', 'Unknown')} - {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg) from e

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
