import pickle # Для сохранения и загрузки индексов
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from tqdm import tqdm


class TFIDFIndexer:
    def __init__(self, save_path: str):
        """
        Для работы с TF-IDF индексом.

        :param save_path: Путь до файла, где сохраняется индекс
        """
        self.vectorizer = TfidfVectorizer()
        self.save_path = save_path
        self.index = None # Здесь будет храниться матрица TF-IDF

    def build_index(self, texts: List[str]):
        """
        Строим TF-IDF для списка текстов и сохраняем его. Сохраняем также векторайзер. На это уходит минут 20-30.

        :param texts: Список текстов для индексации
        """
        for _ in tqdm(range(len(texts))):
            pass # Для отображения прогресса

        self.index = self.vectorizer.fit_transform(texts)
        with open(self.save_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.index), f)

    def load_index(self):
        """Загружаем индекс и векторайзер, который был сохранён."""
        with open(self.save_path, 'rb') as f:
            self.vectorizer, self.index = pickle.load(f)

    def search(self, query: str, top_n: int = 5) -> List[int]:
        """
        Выполняем поиск по запросу и возвращаем индексы наиболее релевантных документов.

        :param query: Текст запроса
        :param top_n: Количество топ результатов
        :return: Список индексов документов
        """
        query_tfidf = self.vectorizer.transform([query])
        scores = (self.index * query_tfidf.T).toarray().flatten()
        ranked_indices = scores.argsort()[::-1][:top_n]
        return ranked_indices

    def get_vector(self, doc_index: int) -> List[float]:
        """
        Возвращаем TF-IDF вектор для документа по его индексу.

        :param doc_index: Индекс документа
        :return: TF-IDF вектор документа
        """

        vector = self.index[doc_index].toarray().flatten()  # Получаем вектор и приводим к одномерному массиву
        return vector
