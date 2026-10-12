import json
import logging

import requests
from google import genai
from google.genai import types

import config
from helper import get_token_count
from retrieval import search

logger = logging.getLogger(__name__)

gemini_client = None
if config.GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set; only Ollama generation will work")

RAG_PROMPT_TEMPLATE = """
You are an AI assistant answering questions about {topic}.
Answer using only the retrieved documents below.
Cite the documents you use inline as [Document N].
If the retrieved documents don't contain the answer, say that you don't know.
{history}
RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}

YOUR ANSWER (accurate, well-structured, with [Document N] citations):
"""

# Number of previous chat messages included in the prompt
HISTORY_MESSAGES = 6
HISTORY_MESSAGE_CHARS = 1000


def _format_history(history):
    if not history:
        return ""
    lines = []
    for message in history[-HISTORY_MESSAGES:]:
        role = "User" if message["role"] == "user" else "Assistant"
        content = str(message["content"])[:HISTORY_MESSAGE_CHARS]
        lines.append(f"{role}: {content}")
    return "\nCONVERSATION SO FAR:\n" + "\n".join(lines) + "\n"


def _format_document(index, hit):
    source = hit["_source"]
    metadata = source.get("metadata") or {}
    header = f"[Document {index} - {source.get('content_type', 'unknown')}]"
    if metadata.get("caption"):
        header += f"\nCaption: {metadata['caption']}"
    return f"{header}\n{source.get('content', '')}"


def build_prompt(query, results, history=None):
    """
    Build the RAG prompt, fitting retrieved documents into MAX_CONTEXT_TOKENS.

    Documents are added in rank order until the budget is used up. The
    question and instructions are never trimmed.
    """
    budget = config.MAX_CONTEXT_TOKENS
    used = 0
    documents = []
    for hit in results:
        text = _format_document(len(documents) + 1, hit)
        tokens = get_token_count(text)
        if used + tokens > budget:
            if documents:
                break
            # The best document alone is over budget: keep a proportional prefix
            text = text[: int(len(text) * budget / tokens)] + "\n...[truncated]"
            tokens = budget
        documents.append(text)
        used += tokens

    if len(documents) < len(results):
        logger.info(
            "Context budget used %d of %d retrieved documents",
            len(documents),
            len(results),
        )

    return RAG_PROMPT_TEMPLATE.format(
        topic=config.DOCUMENT_TOPIC,
        history=_format_history(history),
        context="\n\n---\n\n".join(documents),
        question=query,
    )


def _gemini_config():
    return types.GenerateContentConfig(
        temperature=config.TEMPERATURE,
        top_p=0.95,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
    )


GEMINI_NOT_CONFIGURED = "Gemini is not configured. Set GEMINI_API_KEY in your .env file."
GEMINI_ERROR = "Sorry, Gemini failed to generate a response. Check the server logs."
OLLAMA_ERROR = (
    "Sorry, Ollama failed to generate a response. Make sure Ollama is running "
    f"and the '{config.OLLAMA_MODEL}' model is pulled."
)


def _stream_gemini(prompt_text, model_name):
    if not gemini_client:
        yield GEMINI_NOT_CONFIGURED
        return

    try:
        response_generator = gemini_client.models.generate_content_stream(
            model=model_name,
            contents=prompt_text,
            config=_gemini_config(),
        )
        for chunk in response_generator:
            if getattr(chunk, "text", None):
                yield chunk.text
    except Exception:
        logger.exception("Gemini streaming generation failed")
        yield GEMINI_ERROR


def _complete_gemini(prompt_text, model_name):
    if not gemini_client:
        return GEMINI_NOT_CONFIGURED

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config=_gemini_config(),
        )
        return response.text or "Response received but contained no text."
    except Exception:
        logger.exception("Gemini generation failed")
        return GEMINI_ERROR


def generate_with_gemini(prompt_text, model_name=None, stream=False):
    """Generate a response using the Gemini API."""
    model_name = model_name or config.GEMINI_MODEL
    if stream:
        return _stream_gemini(prompt_text, model_name)
    return _complete_gemini(prompt_text, model_name)


def _stream_ollama(url, data):
    try:
        response = requests.post(
            url, json=data, stream=True, timeout=(10, config.OLLAMA_TIMEOUT)
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    if "response" in chunk:
                        yield chunk["response"]
                except json.JSONDecodeError:
                    continue
    except Exception:
        logger.exception("Ollama streaming generation failed")
        yield OLLAMA_ERROR


def _complete_ollama(url, data):
    try:
        response = requests.post(url, json=data, timeout=(10, config.OLLAMA_TIMEOUT))
        response.raise_for_status()
        return response.json().get("response", "No response generated")
    except Exception:
        logger.exception("Ollama generation failed")
        return OLLAMA_ERROR


def _ollama_context_size(prompt_text):
    """Smallest power of two that fits the prompt plus the answer (min 4096)."""
    needed = int(get_token_count(prompt_text) * 1.2) + config.MAX_OUTPUT_TOKENS
    size = 4096
    while size < needed:
        size *= 2
    return size


def generate_with_ollama(prompt_text, model_name=None, stream=False):
    """Generate a response using a local Ollama model."""
    url = f"{config.OLLAMA_URL}/api/generate"
    data = {
        "model": model_name or config.OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": stream,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_OUTPUT_TOKENS,
            # Ollama's default context window is smaller than our prompts and
            # silently drops the start of the prompt, so size it explicitly
            "num_ctx": _ollama_context_size(prompt_text),
        },
    }
    if stream:
        return _stream_ollama(url, data)
    return _complete_ollama(url, data)


def _retrieval_query(query, history):
    """
    Follow-ups like "explain the second one" retrieve poorly on their own,
    so prepend the previous user question to give the search some context.
    """
    for message in reversed(history or []):
        if message["role"] == "user":
            return f"{message['content']}\n{query}"
    return query


def _as_stream(message):
    yield message


EMPTY_RESULTS_MESSAGE = (
    "No relevant information found. Please try a different search type "
    "or refine your question. Make sure OpenSearch is running and data "
    "has been ingested with python ingestion.py."
)


def generate_rag_response(
    query,
    search_type="hybrid",
    top_k=None,
    model_type="gemini",
    stream=False,
    history=None,
    use_reranker=True,
):
    """
    Retrieve relevant chunks and generate an answer from them.

    Returns:
        tuple: (answer, results) where answer is a string, or a generator of
        text chunks when stream=True, and results are the retrieved hits.
    """
    try:
        results = search(
            _retrieval_query(query, history),
            search_type=search_type,
            top_k=top_k,
            use_reranker=use_reranker,
        )
    except Exception:
        logger.exception("Retrieval failed")
        message = "Sorry, retrieval failed. Check that OpenSearch and Ollama are running."
        return (_as_stream(message) if stream else message), []

    if not results:
        message = EMPTY_RESULTS_MESSAGE
        return (_as_stream(message) if stream else message), []

    prompt_text = build_prompt(query, results, history)
    if model_type == "gemini":
        answer = generate_with_gemini(prompt_text, stream=stream)
    else:
        answer = generate_with_ollama(prompt_text, stream=stream)
    return answer, results


if __name__ == "__main__":
    answer, results = generate_rag_response(
        "How does RAG work?", "hybrid", 3, "gemini", stream=True
    )
    print("Response: ", end="", flush=True)
    for chunk in answer:
        print(chunk, end="", flush=True)
    print(f"\n\n({len(results)} documents retrieved)")
