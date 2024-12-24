import pandas as pd
from preprocessing import preprocess_for_tfidf, preprocess_for_bert
from tfidf_index import TFIDFIndexer
from bert_index import BERTIndexer
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SearchAndLoad:
    def __init__(self, tfidf_path: str, bert_path: str, data_path: str):
        self.data_path = data_path
        self.tfidf_indexer = TFIDFIndexer(tfidf_path)
        self.bert_indexer = BERTIndexer(bert_path)
        self.documents = []

    def load_data(self):
        """Загружаем данные из CSV файла и сохраняем в память."""
        df = pd.read_csv(self.data_path)
        self.documents = df['Текст песни'].tolist()

    def build_indexes(self):
        """Строим оба индекса и сохраняем их."""
        # Предобработка для TF-IDF
        processed_texts_tfidf = [preprocess_for_tfidf(doc) for doc in self.documents]
        self.tfidf_indexer.build_index(processed_texts_tfidf)

        # Предобработка для BERT
        processed_texts_bert = [preprocess_for_bert(doc) for doc in self.documents]
        self.bert_indexer.build_index(processed_texts_bert)

    def search(self, query: str, index_type: str = 'tf-idf', top_n: int = 5) -> List[str]:
        """
        Выполняем поиск по заданному индексу.

        :param query: Запрос для поиска
        :param index_type: Тип индексации ('tf-idf' или 'bert')
        :param top_n: Количество возвращаемых результатов
        :return: Индексы релевантных документов
        """

        if index_type == 'tf-idf':
            if self.tfidf_indexer.index is None:  # Загружаем индексы, если они не загружены
                self.tfidf_indexer.load_index()

            processed_query = preprocess_for_tfidf(query)
            return self.tfidf_indexer.search(processed_query, top_n)
        elif index_type == 'bert':
            if self.bert_indexer.embeddings is None:  # Загружаем индексы, если они не загружены
                self.bert_indexer.load_index()

            processed_query = preprocess_for_bert(query)
            return self.bert_indexer.search(processed_query, top_n)

    def get_vector(self, doc_index: int, index_type: str = 'tf-idf') -> np.ndarray:
        """
        Получаем векторное представление для заданного документа по указанному типу индекса.

        :param doc_index: Индекс документа для векторизации
        :param index_type: Тип индексации ('tf-idf' или 'bert')
        :return: Векторное представление документа
        """
        if index_type == 'tf-idf':
            return self.tfidf_indexer.get_vector(doc_index)
        elif index_type == 'bert':
            return self.bert_indexer.get_embedding(doc_index)


    def pad_vector(self, vector: np.ndarray, target_dimension: int) -> np.ndarray:
        """Дополняем вектор нулями до целевой размерности."""
        if vector.shape[0] < target_dimension:
            padding = np.zeros((target_dimension - vector.shape[0],))
            return np.concatenate((vector, padding))
        return vector

    def get_relevance_score(self, query: str, doc_index: int, index_type: str = 'tf-idf') -> float:
        """
        Вычисляем оценку релевантности (косинусное сходство) для заданного документа и запроса.

        :param query: Запрос для поиска.
        :param doc_index: Индекс документа.
        :param index_type: Тип индекса ('tf-idf' или 'bert').
        :return: Оценка релевантности.
        """
        if index_type == 'tf-idf':
            query_vector = self.tfidf_indexer.vectorizer.transform([preprocess_for_tfidf(query)]).toarray()
            doc_vector = self.tfidf_indexer.get_vector(doc_index)
        elif index_type == 'bert':
            query_inputs = self.bert_indexer.tokenizer(query, return_tensors='pt', truncation=True, padding=True)
            query_outputs = self.bert_indexer.model(**query_inputs)
            query_vector = query_outputs.pooler_output.detach().numpy().flatten()
            doc_vector = self.bert_indexer.get_embedding(doc_index)

        # Косинусное сходство
        similarity = cosine_similarity(query_vector.reshape(1, -1), doc_vector.reshape(1, -1))
        return similarity[0][0]