# Multimodal RAG: Chat with PDFs (Text, Tables & Figures)

A Retrieval-Augmented Generation (RAG) app that answers questions about PDFs using their **text, tables and figures**, with **cited answers** and a **sources panel** showing exactly what each answer was based on.

Search and embeddings run locally (OpenSearch + Ollama). Answers come from **Google Gemini** (fast, cloud) or a **local Ollama model** (private, offline).

<!-- TODO: add demo -->
<!-- ![Demo](docs/demo.gif) -->

## Features

- **Multimodal ingestion.** PDFs are parsed once with `unstructured` (`hi_res`). Figures and tables get LLM-generated descriptions, so a question like *"what does the architecture diagram show?"* can find them.
- **Hybrid search done right.** BM25 keyword search and k-NN semantic search (cosine), merged with **Reciprocal Rank Fusion** so neither score scale dominates.
- **Cross-encoder reranking.** The top 20 candidates are reranked with `BAAI/bge-reranker-base`, and the best few go to the LLM (toggleable).
- **Cited answers.** The model cites `[Document N]`. The Sources panel shows each chunk's file, page, type and score, **renders retrieved tables**, and shows **retrieved figures in a gallery**.
- **Chat with follow-ups.** Conversation history is used for both retrieval and the answer.
- **PDF upload in the UI.** Add documents from the browser. Re-uploading a file replaces its old chunks instead of duplicating them.
- **Gemini or Ollama**, switchable per question.
- **Robust on modest hardware.** Context is trimmed to a token budget (the question is never cut), Ollama's context window is sized to fit the prompt, and Gemini rate limits are retried automatically.
- **Configurable.** Every setting can be overridden in `.env`, including `DOCUMENT_TOPIC`, so the app works for any subject.

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Gradio 6 (chat, sources panel, gallery, upload) |
| PDF parsing | `unstructured` (`hi_res`, table structure, image extraction) |
| Embeddings | `nomic-embed-text` via Ollama (`/api/embed`, batched) |
| Vector + keyword search | OpenSearch 2.11 (HNSW cosine k-NN + BM25) |
| Reranking | `sentence-transformers` cross-encoder (`bge-reranker-base`) |
| LLMs | Google Gemini (`google-genai`) · Ollama (`llama3.2` / `llama3`) |
| Deployment | Docker / Docker Compose |

## Architecture

```
INGESTION
PDF ──► unstructured (hi_res, one parse)
          ├─ figures ──► Gemini description ──────┐
          ├─ tables  ──► Gemini / Ollama summary ─┼─► nomic-embed-text ──► OpenSearch
          └─ text    ──► chunk_by_title ──────────┘                       (BM25 + k-NN)

QUESTION ANSWERING
Question (+ previous question for follow-ups)
   ──► keyword search ─┐
   ──► semantic search ┴─► Reciprocal Rank Fusion ──► cross-encoder rerank ──► top-k
   ──► prompt (token-budgeted context + chat history) ──► Gemini / Ollama
   ──► streamed answer with [Document N] citations + sources, tables, figures
```

## Getting Started

### Prerequisites
- **Python 3.12+** (developed on 3.14)
- **Docker Desktop** (for OpenSearch)
- **[Ollama](https://ollama.com/download)** installed on your machine
- A **[Gemini API key](https://aistudio.google.com/apikey)**. It's required to describe figures during ingestion, and optional for answering.
- For `hi_res` PDF parsing: [Poppler](https://poppler.freedesktop.org/) and [Tesseract](https://github.com/tesseract-ocr/tesseract)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Copy the example config and add your Gemini API key:
```bash
cp .env.example .env
```

### 3. Start OpenSearch
```bash
docker compose up -d opensearch
```
Check it's running by opening http://localhost:9200. You should see JSON with `"cluster_name"`.

### 4. Pull the Ollama models
```bash
ollama pull nomic-embed-text
```
```bash
ollama pull llama3.2
```

### 5. Ingest documents
Put one or more PDFs in `files/`. For example, download the demo paper ([arXiv:2506.18027](https://arxiv.org/pdf/2506.18027)) and save it there. Then ingest every PDF in the folder. You can also skip this step and upload PDFs from the app's **Add documents** tab.
```bash
python ingestion.py
```

### 6. Run the app
```bash
python app.py
```
Open http://127.0.0.1:7860, ask a question, and check the **Sources** panel.

### Full Docker setup (16 GB+ RAM recommended)
To run everything in containers (OpenSearch, Ollama with automatic model download, and the app):
```bash
docker compose up --build
```
⚠️ This starts several memory-heavy services at once. On machines with 8 GB of RAM it can crash Docker. Use the local setup above instead.

## Usage

### Ingestion CLI
```bash
python ingestion.py [PDF ...] [--reset] [--index NAME] [--describe-with gemini|ollama|none]
```
| Option | Effect |
|---|---|
| `PDF ...` | Files to ingest (default: every PDF in `files/`) |
| `--reset` | Delete and recreate the index first |
| `--index` | Target index name (default: `localrag`) |
| `--describe-with` | Model used to describe tables and figures. Figures require Gemini; `ollama` saves Gemini quota |

### App settings
Under **Settings** in the chat tab, you can choose:
- **Search method:** keyword, semantic or hybrid
- **AI model:** Gemini or Ollama
- **Documents to use**
- **Reranking:** on or off
- **Streaming:** on or off

## Configuration

All settings live in [`config.py`](config.py) and can be overridden in `.env` (see [`.env.example`](.env.example)).

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Gemini access (figure descriptions, Gemini answers) |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model |
| `OLLAMA_MODEL` | `llama3` | Local chat model (`llama3.2` recommended on laptops) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `INDEX_NAME` | `localrag` | OpenSearch index |
| `TOP_K` | `5` | Documents passed to the LLM |
| `CANDIDATES` | `20` | Results fetched before fusion and reranking |
| `MAX_CONTEXT_TOKENS` | `6000` | Token budget for retrieved context |
| `TEMPERATURE` | `0.2` | LLM temperature |
| `OLLAMA_TIMEOUT` | `600` | Seconds to wait for Ollama output |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder for reranking |
| `DOCUMENT_TOPIC` | `Retrieval-Augmented Generation (RAG)` | Subject of your documents, used in prompts |

### Running on a low-memory laptop (8 GB RAM, no GPU)
Local models run on the CPU, so answers can take a minute or more. These settings help:
```
OLLAMA_MODEL=llama3.2
MAX_CONTEXT_TOKENS=2000
TOP_K=3
```
For the fastest answers, use **Gemini**.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Failed to connect to OpenSearch` | OpenSearch isn't running. Start it with `docker compose up -d opensearch`. If Docker shows "Engine running" but containers don't respond, quit Docker Desktop, run `wsl --shutdown`, and reopen it |
| `429 RESOURCE_EXHAUSTED` from Gemini | Free-tier quota reached (per minute or per day). Wait, enable billing, or use `--describe-with ollama` and the Ollama model |
| `Ollama failed to generate a response` | The model is still loading or too slow for your machine. Use `llama3.2`, lower `MAX_CONTEXT_TOKENS`, and make sure Ollama is running (`ollama list`) |
| App opens on port 7861 | An older copy of the app is still running on 7860. Stop it with `Ctrl+C` in its terminal |
| First answer is very slow | The reranker model downloads once (about 1 GB). Untick **Rerank results** to skip it |

## Project Structure

```
├── app.py            # Gradio chat UI, sources panel, figure gallery, PDF upload
├── config.py         # Central configuration (overridable via .env)
├── ingestion.py      # Index creation + PDF ingestion pipeline (CLI and ingest_pdf())
├── chunking.py       # Figure/table descriptions (with Gemini retries), text chunking
├── retrieval.py      # Keyword, semantic, hybrid (RRF) search + reranking
├── generation.py     # Prompt building, Gemini/Ollama generation
├── helper.py         # Embeddings, OpenSearch client, token counting
├── files/            # PDFs to ingest
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Acknowledgements

This project began from the Multimodal RAG example in the open-source **awesome-genai-apps** collection, covered in these tutorials:
[Multimodal RAG Overview](https://www.youtube.com/watch?v=kcn6uI87nGc&list=PL0x86ZW374m3uIp_WWOg-jjf-EyXr5KEn) · [PDF Parsing for Multimodal RAG](https://www.youtube.com/watch?v=LIus-y2bJH4&list=PL0x86ZW374m3uIp_WWOg-jjf-EyXr5KEn).

It was then extended with:
- Hybrid search with Reciprocal Rank Fusion, and cross-encoder reranking
- A chat UI with a sources panel, a figure gallery and PDF upload
- Token-budgeted prompts, central configuration, Docker support, and fixes for reliability and performance

## References
- Paper used for the demo: [arXiv:2506.18027](https://arxiv.org/abs/2506.18027)
- [nomic-embed-text](https://www.ollama.com/library/nomic-embed-text) · [llama3.2](https://www.ollama.com/library/llama3.2) · [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- [OpenSearch hybrid search](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) · [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
