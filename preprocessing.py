import re
import string
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Инициализация лемматизатора и скачивание штук для токенайзера. Последние дв строчки этого мини-блока предложила
# сама командная строка, так как не видела punkt
morph = pymorphy2.MorphAnalyzer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Загрузка стоп-слов для русского языка
stop_words = set(stopwords.words('russian'))


def preprocess_for_tfidf(text: str) -> str:
    """
    Предобрабатываем текст для индексации с помощью TF-IDF:
    приводим к нижнему регистру, удаляем пунктуацию, стоп-слова, выполняем лемматизацию.
    Для TF-IDF мы подсчитываем именно вхождение слов. В данном случае важна лемма, пунктуация будет мешать.
    Стоп-слова удалены по причине своей частотности. Для TF-IDF не самый лучший вариант.

    :param text: Исходный текст
    :return: Предобработанный текст
    """
    if not isinstance(text, str):
        text = str(text)  # Преобразуем в строку, если это не строка, так как иногда может вылезать какая-то ошибка

    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление пунктуации
    text = re.sub(f'[{string.punctuation}]', ' ', text)

    # Токенизация текста
    words = word_tokenize(text, language='russian')

    # Лемматизация и удаление стоп-слов
    words = [morph.parse(word)[0].normal_form for word in words if word not in stop_words]

    return ' '.join(words)


def preprocess_for_bert(text: str) -> str:
    """
    Минимальная предобработка для BERT — убираем лишние пробелы. Здесь нужна пунктуация, так как важен контекст.
    Именно поэтому предобработка совсем минимальная. Контекст и каждая мелкая деталь — наше всё.

    :param text: Исходный текст
    :return: Предобработанный текст
    """
    if not isinstance(text, str):
        text = str(text)  # Преобразуем в строку, если это не строка

    return text.strip()  # Убираем лишние пробелы