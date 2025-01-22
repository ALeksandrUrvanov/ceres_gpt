import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
from functools import lru_cache
from typing import List
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import (
    key, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    GPT_MODEL, TEMPERATURE, MAX_TOKENS, DEFAULT_SIMILAR_DOCS_COUNT
)


class VineyardAssistant:
    """
    Класс для обработки запросов о виноградарстве с использованием векторного поиска и GPT.
    """
    def __init__(self):
        """Инициализация ассистента с настройкой всех необходимых компонентов."""
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
        similar_docs = retriever.get_relevant_documents(query)
        cleaned_docs = [
            Document(page_content=self.clean_text(doc.page_content))
            for doc in similar_docs
        ]
        return cleaned_docs

    async def process_query(self, user_query: str) -> str:
        """Асинхронная обработка запроса пользователя."""
        try:
            if not user_query or not isinstance(user_query, str):
                raise ValueError("Query must be a non-empty string")

            processed_query = self.preprocess_query(user_query)
            similar_docs = self.get_similar_documents(processed_query)
            context = "\n\n".join(doc.page_content for doc in similar_docs)

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": """Вы - эксперт по виноградарству компании Ceres Pro.
                            Ваша задача - отвечать на вопросы пользователя. 
                            Если в предоставленном контексте есть информация, используйте её для ответа. 
                            Если информации недостаточно, опирайтесь на собственные знания. 
                            Пишите ответ так, чтобы он был полезен и информативен. 
                            В конце каждого ответа спрашивайте, остались ли у пользователя ещё вопросы."""},
                    {"role": "user", "content": f"Контекст:\n\n{context}\n\nВопрос: {user_query}\n\n"}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз или переформулируйте вопрос."

    async def process_queries_batch(self, queries: List[str]) -> List[str]:
        """Пакетная обработка нескольких запросов."""
        return await asyncio.gather(*(self.process_query(query) for query in queries))
