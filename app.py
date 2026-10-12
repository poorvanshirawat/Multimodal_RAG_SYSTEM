import base64
import html
import io
import logging
import os
import shutil

import gradio as gr
from PIL import Image

import config
from generation import generate_rag_response
from ingestion import ingest_pdf

logger = logging.getLogger(__name__)


def _message_text(message):
    """Gradio 6 sends message content as a list of blocks; flatten it to text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content or "")


def _plain_history(history):
    return [
        {"role": message["role"], "content": _message_text(message)}
        for message in history
    ]


def _decode_image(base64_image):
    try:
        return Image.open(io.BytesIO(base64.b64decode(base64_image)))
    except Exception:
        logger.warning("Could not decode a retrieved image")
        return None


def render_sources(results):
    """Build the Sources panel HTML and the gallery of retrieved figures."""
    if not results:
        return "<p><em>No sources retrieved.</em></p>", []

    parts = []
    gallery = []
    for i, hit in enumerate(results, start=1):
        source = hit["_source"]
        metadata = source.get("metadata") or {}
        content_type = source.get("content_type", "text")
        score = hit.get("rerank_score", hit.get("_score"))

        location = html.escape(metadata.get("filename") or "unknown file")
        if metadata.get("page_number"):
            location += f", p. {metadata['page_number']}"
        score_text = f" · score {score:.3f}" if score is not None else ""

        body = ""
        if content_type == "table" and source.get("table_html"):
            body += f"<div class='source-table'>{source['table_html']}</div>"
        if content_type == "image":
            caption = metadata.get("caption") or ""
            if caption:
                body += f"<p><em>{html.escape(caption)}</em></p>"
            if source.get("base64_image"):
                image = _decode_image(source["base64_image"])
                if image is not None:
                    gallery.append((image, f"Document {i}: {caption}"))

        snippet = html.escape(source.get("content", "")[:500])
        parts.append(
            f"<details><summary><b>[Document {i}]</b> {content_type} · "
            f"{location}{score_text}</summary>{body}<p>{snippet}…</p></details>"
        )

    return "\n".join(parts), gallery


def respond(message, history, search_type, model_type, top_k, use_reranker, stream):
    """Answer a chat message, streaming tokens and updating the sources panel."""
    if not message or not message.strip():
        gr.Warning("Please enter a question.")
        yield history, message, gr.skip(), gr.skip()
        return

    previous = _plain_history(history)
    history = previous + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Retrieving relevant information..."},
    ]
    yield history, "", gr.skip(), gr.skip()

    answer, results = generate_rag_response(
        message,
        search_type=search_type,
        top_k=int(top_k),
        model_type=model_type,
        stream=stream,
        history=previous,
        use_reranker=use_reranker,
    )
    sources_html, gallery = render_sources(results)

    if stream:
        text = ""
        for chunk in answer:
            text += chunk
            history[-1]["content"] = text
            yield history, "", sources_html, gallery
    else:
        history[-1]["content"] = answer
        yield history, "", sources_html, gallery


def upload_pdfs(files, describe_with, progress=gr.Progress()):
    """Save uploaded PDFs to UPLOAD_DIR and ingest them."""
    if not files:
        return "Please choose at least one PDF."

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    log = []
    for i, file_path in enumerate(files):
        destination = os.path.join(config.UPLOAD_DIR, os.path.basename(file_path))
        shutil.copy(file_path, destination)

        def report(status, i=i):
            progress(i / len(files), desc=status)

        try:
            count = ingest_pdf(destination, describe_with=describe_with, progress=report)
            log.append(f"✅ {os.path.basename(destination)}: {count} chunks indexed")
        except Exception as e:
            logger.exception("Ingestion failed for %s", destination)
            log.append(f"❌ {os.path.basename(destination)}: {e}")
    return "\n".join(log)


CSS = """
.source-table { overflow-x: auto; font-size: 0.85em; }
.source-table table { border-collapse: collapse; }
.source-table td, .source-table th { border: 1px solid #8884; padding: 2px 6px; }
"""

with gr.Blocks(title="LocalRAG Q&A System") as demo:
    gr.Markdown("# 📚 LocalRAG Multimodal Q&A")
    gr.Markdown(
        f"Ask questions about your indexed documents ({config.DOCUMENT_TOPIC}). "
        "Answers cite their sources, and retrieved figures and tables are shown alongside."
    )

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=520)
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about your documents...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear conversation")

                with gr.Accordion("Settings", open=False):
                    with gr.Row():
                        search_type = gr.Radio(
                            ["keyword", "semantic", "hybrid"],
                            label="Search Method",
                            value="hybrid",
                            info="Hybrid merges keyword and semantic results",
                        )
                        model_type = gr.Radio(
                            ["gemini", "ollama"],
                            label="AI Model",
                            value="gemini",
                            info=f"Gemini uses Google's API; Ollama uses your local {config.OLLAMA_MODEL}",
                        )
                    with gr.Row():
                        top_k = gr.Slider(
                            1, 15, value=config.TOP_K, step=1, label="Documents to use"
                        )
                        use_reranker = gr.Checkbox(
                            label="Rerank results",
                            value=True,
                            info="More accurate; the first query loads the model",
                        )
                        stream_checkbox = gr.Checkbox(
                            label="Stream Response",
                            value=True,
                            info="See the answer as it's being generated",
                        )

            with gr.Column(scale=2):
                gr.Markdown("### Sources")
                sources = gr.HTML("<p><em>Sources appear here after you ask.</em></p>")
                gallery = gr.Gallery(label="Retrieved figures", columns=2, height=300)

        inputs = [
            query_input,
            chatbot,
            search_type,
            model_type,
            top_k,
            use_reranker,
            stream_checkbox,
        ]
        outputs = [chatbot, query_input, sources, gallery]
        submit_btn.click(respond, inputs=inputs, outputs=outputs, show_progress="minimal")
        query_input.submit(respond, inputs=inputs, outputs=outputs, show_progress="minimal")
        clear_btn.click(
            lambda: ([], "<p><em>Sources appear here after you ask.</em></p>", []),
            outputs=[chatbot, sources, gallery],
        )

        gr.Examples(
            examples=[
                ["How does RAG work?"],
                ["What are the benefits of RAG compared to fine-tuning?"],
                ["Explain the RAG architecture shown in the figures"],
                ["What are common challenges in RAG implementations?"],
            ],
            inputs=[query_input],
        )

    with gr.Tab("Add documents"):
        gr.Markdown(
            "Upload PDFs to add them to the index. Parsing uses unstructured's "
            "`hi_res` strategy, so expect a few minutes per document. Uploading a "
            "file with the same name again replaces its old chunks."
        )
        pdf_files = gr.File(
            label="PDF files", file_types=[".pdf"], file_count="multiple", type="filepath"
        )
        describe_with = gr.Radio(
            ["gemini", "ollama", "none"],
            value="gemini",
            label="Describe images and tables with",
            info="Image descriptions need Gemini; Ollama only describes tables",
        )
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_status = gr.Textbox(label="Status", lines=6)
        ingest_btn.click(upload_pdfs, inputs=[pdf_files, describe_with], outputs=ingest_status)

    gr.Markdown(
        """
    ### 📘 How to use this system
    1. **Ask a question**. Follow-up questions use the conversation so far.
    2. Under **Settings**, choose a search method:
       - **Keyword**: BM25 text search
       - **Semantic**: meaning-based search using embeddings
       - **Hybrid**: both, merged with Reciprocal Rank Fusion
    3. Pick **Gemini** (API key required) or **Ollama** (local).
    4. Check **Sources** to see exactly which chunks, tables and figures the answer used.
    """
    )

if __name__ == "__main__":
    demo.queue().launch(
        theme="soft",
        css=CSS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    )
