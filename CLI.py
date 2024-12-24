import click
import sys
import time
from search import SearchAndLoad

TFIDF_INDEX_PATH = 'indexes/tfidf_index.pkl'
BERT_INDEX_PATH = 'indexes/bert_index.pkl'
DATA_PATH = 'data/songs_data_corpus.csv'

# Создаем объект SearchAndLoad для использования в командах
search_and_load = SearchAndLoad(tfidf_path=TFIDF_INDEX_PATH, bert_path=BERT_INDEX_PATH, data_path=DATA_PATH)


@click.group()
def cli():
    """Главный интерфейс командной строки для работы с индексами и поиском."""
    pass

@click.command()
def greet():
    """Как сделать поиск."""
    click.echo("Чтобы выполнить поиск, используйте команду:")
    click.echo("python cli.py search-cli <ваш запрос> --index <tf-idf|bert>")
    click.echo("Вы также можете создать индексы с помощью команды 'build_indexes-cli'.")

@click.command()
def build_indexes_cli():
    """Команда для создания TF-IDF и BERT индексов."""
    click.echo("Загрузка данных для индексации...")
    search_and_load.load_data()

    click.echo("Создание индексов...")
    search_and_load.build_indexes()

    click.echo(f"Индексы успешно созданы и сохранены.")


@click.command()
@click.argument('query', type=str)
@click.option('--index', type=click.Choice(['tf-idf', 'bert'], case_sensitive=False), required=True,
              help="Тип индекса: 'tf-idf' или 'bert'")
def search_cli(query: str, index: str):
    """
    Команда для поиска по запросу с использованием указанного индекса.

    :param query: Запрос для поиска.
    :param index: Тип индекса ('tf-idf' или 'bert').
    """
    click.echo(f"Поиск по запросу: {query} с использованием индекса {index}...")

    start_time = time.time()  # Время начала поиска

    # Выполняем поиск
    tfidf_results = []
    bert_results = []

    if index == 'tf-idf':
        tfidf_results = search_and_load.search(query=query, index_type='tf-idf')
    elif index == 'bert':
        bert_results = search_and_load.search(query=query, index_type='bert')

    elapsed_time = time.time() - start_time # Затраченное время

    # Загружаем тексты документов для вывода
    search_and_load.load_data()
    all_texts = search_and_load.documents

    if index == 'tf-idf' and len(tfidf_results) > 0:
        click.echo("Результаты поиска (TF-IDF):")
        for idx in range(len(tfidf_results)):
            doc_index = tfidf_results[idx]
            click.echo(f"\n--- Результат {idx + 1} ---\n{all_texts[doc_index]}")
            relevance_score = search_and_load.get_relevance_score(query, doc_index, index_type='tf-idf')
            click.echo(f"Оценка релевантности: {relevance_score:.4f}")


    elif index == 'bert' and len(bert_results) > 0:
        click.echo("Результаты поиска (BERT):")
        for idx in range(len(bert_results)):
            doc_index = bert_results[idx]
            click.echo(f"\n--- Результат {idx + 1} ---\n{all_texts[doc_index]}")
            doc_index = bert_results[idx]
            relevance_score = search_and_load.get_relevance_score(query, doc_index, index_type='bert')
            click.echo(f"Оценка релевантности: {relevance_score:.4f}")

    else:
        click.echo("Ничего не найдено.")

    click.echo(f"\nВремя выполнения поиска: {elapsed_time:.2f} секунд.")

# Добавляем команды в главный интерфейс
cli.add_command(greet)
cli.add_command(build_indexes_cli)
cli.add_command(search_cli)

if __name__ == '__main__':
    # Проверяем, если скрипт запущен без параметров, вызываем команду greet по умолчанию.
    if len(sys.argv) == 1:
        greet()
    else:
        cli()