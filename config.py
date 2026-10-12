"""Central configuration. Every value can be overridden in .env."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# Models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
# Seconds to wait for Ollama's next output. The first token can be slow on a
# CPU because the model must load and read the whole prompt first.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# nomic-embed-text was trained with task prefixes; set both to "" for other models
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")
EMBED_DOCUMENT_PREFIX = os.getenv("EMBED_DOCUMENT_PREFIX", "search_document: ")

# OpenSearch
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
INDEX_NAME = os.getenv("INDEX_NAME", "localrag")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
# How many candidates each search returns before fusion / reranking
CANDIDATES = int(os.getenv("CANDIDATES", "20"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
# Token budget for retrieved context in the prompt (the question is never trimmed)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

# What the indexed documents are about; used in the LLM prompts
DOCUMENT_TOPIC = os.getenv("DOCUMENT_TOPIC", "Retrieval-Augmented Generation (RAG)")

# Ingestion
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "files")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
