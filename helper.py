import logging
from functools import lru_cache

import requests
from opensearchpy import OpenSearch

import config

logger = logging.getLogger(__name__)


def get_embeddings(texts, kind="document", model=None, batch_size=32):
    """
    Embed many texts with Ollama's /api/embed endpoint, in batches.

    Args:
        texts (list[str]): Texts to embed.
        kind (str): "document" for indexed chunks, "query" for search queries.
            Selects the task prefix that nomic-embed-text was trained with.
        model (str): Ollama embedding model. Defaults to config.EMBEDDING_MODEL.
        batch_size (int): Number of texts sent per request.

    Returns:
        list[list[float]]: One embedding vector per input text.
    """
    model = model or config.EMBEDDING_MODEL
    prefix = (
        config.EMBED_QUERY_PREFIX if kind == "query" else config.EMBED_DOCUMENT_PREFIX
    )
    url = f"{config.OLLAMA_URL}/api/embed"

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        response = requests.post(
            url, json={"model": model, "input": batch}, timeout=300
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Error fetching embeddings: {response.status_code}, {response.text}"
            )
        embeddings.extend(response.json()["embeddings"])
    return embeddings


def get_embedding(text, kind="query", model=None):
    """Embed a single text. See get_embeddings."""
    return get_embeddings([text], kind=kind, model=model)[0]


@lru_cache(maxsize=1)
def get_opensearch_client():
    """Return a shared OpenSearch client, created and checked once per process."""
    client = OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    if not client.ping():
        # Not cached on failure, so the next call retries the connection
        get_opensearch_client.cache_clear()
        raise ConnectionError(
            f"Failed to connect to OpenSearch at "
            f"{config.OPENSEARCH_HOST}:{config.OPENSEARCH_PORT}."
        )

    info = client.info()
    logger.info(
        "Connected to OpenSearch cluster %s (version %s)",
        info["cluster_name"],
        info["version"]["number"],
    )
    return client


@lru_cache(maxsize=1)
def _get_encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def get_token_count(text):
    """
    Approximate the token count of text.

    Uses tiktoken's cl100k_base encoding. It is not the exact tokenizer of
    Gemini or llama3, but it is close enough for budgeting prompt context.
    """
    return len(_get_encoding().encode(text))


if __name__ == "__main__":
    try:
        embedding = get_embedding("This is a test prompt.")
        print(f"Embedding dimension: {len(embedding)}")
    except Exception as e:
        print("Failed to get embedding:", e)
