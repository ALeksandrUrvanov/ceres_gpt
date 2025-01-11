# Ceres GPT - Виноградарский Ассистент

## Описание
Ceres GPT - это интеллектуальный ассистент-агроном, реализованный в виде консольного приложения и Telegram бота. Система предоставляет экспертные консультации по всем аспектам виноградарства, используя современные технологии обработки естественного языка и машинного обучения.

## Основные функции

### Общие возможности
- Контекстный поиск информации
- Интеллектуальная обработка запросов
- Экспертные ответы по виноградарству
- Многоязычная поддержка

### Telegram бот
- Интерактивный интерфейс с командами
- Индикация набора текста
- Автоматическое разделение длинных сообщений
- Обработка ошибок с понятными сообщениями
- Логирование работы бота

## Технологии
- Python 3.8+
- OpenAI API (GPT-4-mini)
- aiogram 3.x
- LangChain
- Hugging Face Transformers
- FAISS
- AsyncIO

# Установка зависимостей
pip install aiogram openai langchain faiss-cpu
pip install -r requirements.txt

### Конфигурация
`config.py`:

bot_token = "ВАШ_ТОКЕН_TELEGRAM"
key = "ВАШ_КЛЮЧ_OPENAI"

## Использование

### Запуск консольной версии
python main.py

### Запуск Telegram бота
python bot.py

## Команды Telegram бота
- `/start` - Начало работы с ботом
- `/help` - Получение справки
- `/exit` - Завершение диалога

## Особенности реализации
- Асинхронная обработка запросов
- Многопоточная обработка тяжелых вычислений
- Автоматическое разделение длинных ответов (>4000 символов)
- Система логирования с отслеживанием ошибок
- Векторное хранилище для быстрого поиска
  
`markdown
## Структура проекта
ceres_gpt/
├── main.py          # Консольное приложение
├── bot.py           # Telegram бот
├── config.py        # Конфигурация
├── data/            # Данные для обучения
│   ├── *.txt        # Текстовые файлы с информацией
│   └── faiss_index/ # Векторное хранилище
└── requirements.txt
`

## Обработка ошибок
- Автоматическое восстановление при сбоях
- Информативные сообщения об ошибках
- Система логирования

# Логирование
- Уровень: INFO
- Отслеживание запуска/остановки бота
- Логирование ошибок обработки сообщений

## Безопасность
- Безопасное хранение токенов
- Обработка некорректных запросов
- Защита от переполнения сообщений




