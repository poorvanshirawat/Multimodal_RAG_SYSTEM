import argparse
import glob
import logging
import os

import config

logger = logging.getLogger(__name__)


def ensure_index(client, index_name, reset=False):
    """
    Create the OpenSearch index with a vector-search mapping if it doesn't exist.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index
        reset: Delete and recreate the index, removing all ingested documents
    """
    if client.indices.exists(index=index_name):
        if not reset:
            logger.info("Using existing index '%s'", index_name)
            return
        logger.info("Deleting existing index '%s' (--reset)", index_name)
        client.indices.delete(index=index_name)

    from helper import get_embedding

    dimension = len(get_embedding("Sample text for dimension detection", kind="document"))
    logger.info("Using embedding dimension: %d", dimension)

    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    # space_type must be set per field; the index-level setting is ignored
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
                "base64_image": {"type": "binary", "doc_values": False},
                "table_html": {"type": "text", "index": False},
                "metadata": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "caption": {"type": "text"},
                        "image_text": {"type": "text"},
                    }
                },
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    logger.info("Created index '%s' with vector search capabilities", index_name)


def prepare_chunks_for_ingestion(chunks):
    """
    Prepare chunks for ingestion by adding embeddings and token counts.

    Args:
        chunks: List of chunks to prepare

    Returns:
        List of prepared chunks ready for ingestion
    """
    from helper import get_embeddings, get_token_count

    chunks = [chunk for chunk in chunks if chunk.get("content")]
    logger.info("Embedding %d chunks", len(chunks))
    embeddings = get_embeddings([chunk["content"] for chunk in chunks], kind="document")

    prepared_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        doc = {
            "content": chunk["content"],
            "content_type": chunk.get("content_type", "text"),
            "token_count": get_token_count(chunk["content"]),
            "embedding": embedding,
            "metadata": {
                "filename": chunk.get("filename", ""),
                "page_number": chunk.get("page_number"),
                "caption": chunk.get("caption", ""),
                "image_text": chunk.get("image_text", ""),
            },
        }

        if doc["content_type"] == "image" and chunk.get("base64_image"):
            doc["base64_image"] = chunk["base64_image"]

        if doc["content_type"] == "table" and chunk.get("table_as_html"):
            doc["table_html"] = chunk["table_as_html"]

        prepared_chunks.append(doc)

    logger.info("Prepared %d chunks for ingestion", len(prepared_chunks))
    return prepared_chunks


def ingest_chunks_into_opensearch(client, index_name, chunks):
    """
    Bulk-ingest prepared chunks into OpenSearch.

    Returns:
        Number of successfully ingested documents
    """
    from opensearchpy.helpers import bulk

    actions = ({"_index": index_name, "_source": chunk} for chunk in chunks)
    successful, errors = bulk(
        client, actions, chunk_size=100, raise_on_error=False, refresh=True
    )
    if errors:
        logger.warning("%d documents failed to ingest: %s", len(errors), errors[:3])
    logger.info("Ingestion complete: %d successful, %d failed", successful, len(errors))
    return successful


def delete_file_from_index(client, index_name, filename):
    """Remove previously ingested chunks of a file so re-ingesting replaces them."""
    response = client.delete_by_query(
        index=index_name,
        body={"query": {"term": {"metadata.filename": filename}}},
        refresh=True,
    )
    if response.get("deleted"):
        logger.info("Removed %d old chunks of '%s'", response["deleted"], filename)


def ingest_pdf(pdf_path, index_name=None, describe_with="gemini", progress=None):
    """
    Parse a PDF, describe its images and tables, chunk its text, and index it all.

    Re-ingesting a file with the same name replaces its previous chunks.

    Args:
        pdf_path: Path to the PDF
        index_name: Target index. Defaults to config.INDEX_NAME
        describe_with: "gemini", "ollama" or "none". Model used to describe
            tables (and images, which need Gemini) so they can be searched
        progress: Optional callable(str) receiving status messages

    Returns:
        Number of chunks ingested
    """
    from unstructured.partition.pdf import partition_pdf

    from chunking import (
        create_semantic_chunks,
        process_images_with_captions,
        process_tables_with_descriptions,
    )
    from helper import get_opensearch_client

    index_name = index_name or config.INDEX_NAME

    def report(message):
        logger.info(message)
        if progress:
            progress(message)

    client = get_opensearch_client()
    ensure_index(client, index_name)

    filename = os.path.basename(pdf_path)
    report(f"Parsing {filename} (this can take a few minutes)...")
    # One hi_res parse feeds images, tables and text chunks
    raw_chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure"],
        extract_image_block_to_payload=True,
    )

    report("Describing images...")
    processed_images = process_images_with_captions(
        raw_chunks, use_gemini=describe_with == "gemini"
    )

    report("Describing tables...")
    processed_tables = process_tables_with_descriptions(
        raw_chunks,
        use_gemini=describe_with == "gemini",
        use_ollama=describe_with == "ollama",
    )

    report("Chunking text...")
    semantic_chunks = create_semantic_chunks(raw_chunks)

    all_chunks = processed_images + processed_tables + semantic_chunks
    report(f"Embedding {len(all_chunks)} chunks...")
    prepared_chunks = prepare_chunks_for_ingestion(all_chunks)

    delete_file_from_index(client, index_name, filename)
    count = ingest_chunks_into_opensearch(client, index_name, prepared_chunks)
    report(f"Ingested {count} chunks from {filename}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into OpenSearch.")
    parser.add_argument(
        "pdfs",
        nargs="*",
        help=f"PDF files to ingest (default: every PDF in {config.UPLOAD_DIR}/)",
    )
    parser.add_argument("--index", default=config.INDEX_NAME, help="Index name")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the index before ingesting",
    )
    parser.add_argument(
        "--describe-with",
        choices=["gemini", "ollama", "none"],
        default="gemini",
        help="Model used to describe images and tables (images need Gemini)",
    )
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(glob.glob(os.path.join(config.UPLOAD_DIR, "*.pdf")))
    if not pdfs:
        parser.error(f"No PDFs given and none found in {config.UPLOAD_DIR}/")

    if args.reset:
        from helper import get_opensearch_client

        ensure_index(get_opensearch_client(), args.index, reset=True)

    total = 0
    for pdf in pdfs:
        total += ingest_pdf(pdf, index_name=args.index, describe_with=args.describe_with)
    print(f"Successfully ingested {total} chunks from {len(pdfs)} file(s)")


if __name__ == "__main__":
    main()
