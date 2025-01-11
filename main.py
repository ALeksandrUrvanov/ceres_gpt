import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import key

class VineyardAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.initialize_vector_store()

    def initialize_vector_store(self):
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

    def load_training_data(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                    documents.extend(self.text_splitter.create_documents([text]))
                except Exception:
                    pass
        return documents

    def preprocess_query(self, query: str) -> str:
        query = query.strip().lower()
        if not any(word in query for word in ['виноград', 'лоза', 'куст', 'сорт', 'урожай', 'обработка']):
            query = f"виноград {query}"
        return query

    def process_query(self, user_query: str) -> str:
        try:
            processed_query = self.preprocess_query(user_query)
            similar_docs = self.vector_store.similarity_search(processed_query, k=5)
            context = "\n\n".join(doc.page_content for doc in similar_docs)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Вы - эксперт по виноградарству. Ваша задача - отвечать на вопросы 
                    пользователя. Если в предоставленном контексте есть информация, используйте её для ответа. 
                    Если информации недостаточно, опирайтесь на собственные знания. 
                    Пишите ответ так, чтобы он был полезен и информативен."""},
                    {"role": "user",
                     "content": f"Контекст:\n\n{context}\n\nВопрос: {user_query}\n\n"}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return "Ошибка при обработке запроса."


def main():
    assistant = VineyardAssistant()
    print("Здравствуйте! Я ваш помощник-агроном. Чем могу помочь?")

    query = input("\nВведите ваш вопрос (или 'выход' для завершения): ").strip()
    if query.lower() in ['выход', 'exit', 'quit']:
        print("\nСпасибо за использование помощника-агронома. Удачи в виноградарстве!")
        return

    response = assistant.process_query(query)
    print("\nОтвет:", response)

    while True:
        query = input("\nОстались ли ещё вопросы? Если нет, введите 'выход': ").strip()
        if query.lower() in ['выход', 'exit', 'quit']:
            print("\nСпасибо за использование помощника-агронома. Удачи в виноградарстве!")
            break

        response = assistant.process_query(query)
        print("\nОтвет:", response)


if __name__ == "__main__":
    main()