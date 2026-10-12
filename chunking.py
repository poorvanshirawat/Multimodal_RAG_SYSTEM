import base64
import logging
import re
import time

import requests

import config

logger = logging.getLogger(__name__)


def _get_gemini_client():
    from google import genai

    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


RETRYABLE_STATUS_CODES = {429, 500, 503}
MAX_ATTEMPTS = 5


def _retry_delay(error, attempt):
    """Use the delay Gemini suggests ("retry in 43.2s"), else exponential backoff."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 1
    return min(5 * 2**attempt, 60)


def _generate_with_retry(gemini_client, **kwargs):
    """
    Call Gemini, waiting and retrying on rate limits (429) and overload (5xx).

    The free tier allows only a few requests per minute, so a document with
    several figures and tables would otherwise hit the limit partway through.
    """
    from google.genai import errors

    for attempt in range(MAX_ATTEMPTS):
        try:
            return gemini_client.models.generate_content(**kwargs)
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS - 1:
                raise
            delay = _retry_delay(e, attempt)
            logger.info(
                "Gemini returned %s; retrying in %.0fs (attempt %d/%d)",
                e.code,
                delay,
                attempt + 2,
                MAX_ATTEMPTS,
            )
            time.sleep(delay)


def _element_metadata(element):
    metadata = getattr(element, "metadata", None)
    return {
        "filename": getattr(metadata, "filename", None) or "",
        "page_number": getattr(metadata, "page_number", None),
    }


def process_images_with_captions(raw_chunks, use_gemini=True):
    """
    Extract images from raw document chunks, identify captions, and generate descriptions.

    Args:
        raw_chunks: List of document elements from unstructured.partition_pdf
        use_gemini: Whether to use Gemini for image captioning (default: True)

    Returns:
        List of dictionaries with image data, captions, and generated descriptions
    """
    from google.genai import types
    from unstructured.documents.elements import FigureCaption, Image

    gemini_client = _get_gemini_client() if use_gemini else None

    processed_images = []
    errors = 0

    for idx, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, Image):
            continue

        image_base64 = getattr(chunk.metadata, "image_base64", None)
        if not image_base64:
            continue

        # Check if next element is a figure caption
        if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
            caption = raw_chunks[idx + 1].text
        else:
            caption = ""

        image_text = chunk.text or ""
        mime_type = getattr(chunk.metadata, "image_mime_type", None) or "image/jpeg"

        image_data = {
            "caption": caption,
            "image_text": image_text,
            "base64_image": image_base64,
            "content": image_text,  # Fallback content
            "content_type": "image",
            **_element_metadata(chunk),
        }

        if use_gemini:
            try:
                prompt = (
                    f"Generate a comprehensive and detailed description of this image from a technical document about {config.DOCUMENT_TOPIC}.\n\n"
                    f"CONTEXT INFORMATION:\n"
                    f"- Caption: {caption or 'None'}\n"
                    f"- Text extracted from image: {image_text or 'No text'}\n\n"
                    f"DESCRIPTION REQUIREMENTS:\n"
                    f"1. Begin with a clear overview of what the image shows (diagram, chart, architecture, etc.)\n"
                    f"2. If it's a diagram or flowchart: describe components, connections, data flow direction, and system architecture\n"
                    f"3. If it's a chart or graph: explain axes, trends, key data points, and significance\n"
                    f"4. Explain technical terminology and abbreviations that appear in the image\n"
                    f"5. Interpret how this visual relates to the document's subject\n"
                    f"6. Include any numerical data, performance metrics, or comparative results shown\n"
                    f"7. Target length: 150-300 words for complex diagrams, 100-150 words for simpler images\n\n"
                    f"Focus on providing information that would be valuable to someone researching {config.DOCUMENT_TOPIC}."
                )

                response = _generate_with_retry(
                    gemini_client,
                    model=config.GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(
                            data=base64.b64decode(image_base64), mime_type=mime_type
                        ),
                        prompt,
                    ],
                )
                image_data["content"] = response.text
            except Exception as e:
                logger.warning("Error generating image description: %s", e)
                errors += 1

        processed_images.append(image_data)

    logger.info("Processed %d images (%d errors)", len(processed_images), errors)
    return processed_images


def process_tables_with_descriptions(raw_chunks, use_gemini=True, use_ollama=False):
    """Process tables with descriptions using Gemini or Ollama"""
    from unstructured.documents.elements import Table

    gemini_client = _get_gemini_client() if use_gemini else None

    processed_tables = []
    errors = 0

    for chunk in raw_chunks:
        if not isinstance(chunk, Table):
            continue

        table_html = getattr(chunk.metadata, "text_as_html", None) or chunk.text
        table_data = {
            "table_as_html": table_html,
            "content": chunk.text,  # Fallback content
            "content_type": "table",
            **_element_metadata(chunk),
        }

        if use_gemini:
            try:
                prompt = (
                    f"Generate a comprehensive and detailed description of the following table from a technical document about {config.DOCUMENT_TOPIC}.\n\n"
                    f"TABLE HTML:\n{table_html}\n\n"
                    f"DESCRIPTION REQUIREMENTS:\n"
                    f"1. Provide an overview of the table's purpose and what it represents in the context of the document.\n"
                    f"2. Explain the significance of each column and row, including any key metrics or data points.\n"
                    f"3. Describe any trends, comparisons, or notable findings presented in the table.\n"
                    f"4. If applicable, explain how this data supports or illustrates the document's concepts.\n"
                    f"5. Target length: 150-300 words.\n"
                    f"6. Do not include information like 'This table shows' or 'The table contains', but rather directly explain the content and significance of the table.\n\n"
                    f"Directly return the summary without additional commentary or preamble."
                )

                response = _generate_with_retry(
                    gemini_client,
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                )
                table_data["content"] = response.text
            except Exception as e:
                logger.warning("Error generating table description with Gemini: %s", e)
                errors += 1

        elif use_ollama:
            try:
                data = {
                    "model": config.OLLAMA_MODEL,
                    "prompt": (
                        "Analyze the following table and provide a detailed summary of its contents, "
                        "including the structure, key data points, and any notable trends or insights. "
                        f"Here is the table in HTML format: {table_html}\n"
                        "Directly analyze the table and provide a detailed summary without any additional text."
                    ),
                    "stream": False,
                    "options": {"num_predict": 1000, "temperature": 0.2},
                }

                response = requests.post(
                    f"{config.OLLAMA_URL}/api/generate", json=data, timeout=300
                )
                response.raise_for_status()
                # Keep the raw table text if the model returned nothing
                table_data["content"] = (
                    response.json().get("response") or table_data["content"]
                )
            except Exception as e:
                logger.warning("Error generating table description with Ollama: %s", e)
                errors += 1

        processed_tables.append(table_data)

    logger.info("Processed %d tables (%d errors)", len(processed_tables), errors)
    return processed_tables


def create_semantic_chunks(raw_chunks):
    """
    Group a document's text into section-sized chunks based on its title structure.

    Tables and images are excluded because they are indexed separately with
    their own generated descriptions; keeping them here would duplicate them.

    Args:
        raw_chunks: List of document elements from unstructured.partition_pdf

    Returns:
        List of semantic chunks
    """
    from unstructured.chunking.title import chunk_by_title
    from unstructured.documents.elements import Image, Table

    text_elements = [
        element
        for element in raw_chunks
        if not isinstance(element, (Table, Image)) and element.text
    ]

    chunks = chunk_by_title(
        text_elements,
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500,
    )

    processed_chunks = [
        {
            "content": chunk.text,
            "content_type": "text",
            **_element_metadata(chunk),
        }
        for chunk in chunks
        if chunk.text.strip()
    ]

    logger.info("Created %d semantic chunks from document", len(processed_chunks))
    return processed_chunks
