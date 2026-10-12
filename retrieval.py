import logging
from functools import lru_cache

import config
from helper import get_embedding, get_opensearch_client

logger = logging.getLogger(__name__)

SOURCE_FIELDS = [
    "content",
    "content_type",
    "token_count",
    "metadata",
    "base64_image",
    "table_html",
]

# Standard constant from the Reciprocal Rank Fusion paper (Cormack et al., 2009)
RRF_K = 60


def keyword_search(query_text, top_k=20):
    """
    Perform BM25 keyword search.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search hits
    """
    try:
        search_query = {
            "size": top_k,
            "query": {"match": {"content": query_text}},
            "_source": SOURCE_FIELDS,
        }
        response = get_opensearch_client().search(
            index=config.INDEX_NAME, body=search_query
        )
        return response["hits"]["hits"]
    except Exception:
        logger.exception("Keyword search failed")
        return []


def semantic_search(query_text, top_k=20):
    """
    Perform k-NN search over the embeddings.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search hits
    """
    try:
        query_embedding = get_embedding(query_text, kind="query")
        search_query = {
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
            "_source": SOURCE_FIELDS,
        }
        response = get_opensearch_client().search(
            index=config.INDEX_NAME, body=search_query
        )
        return response["hits"]["hits"]
    except Exception:
        logger.exception("Semantic search failed")
        return []


def reciprocal_rank_fusion(result_lists, top_k):
    """
    Merge ranked hit lists with Reciprocal Rank Fusion.

    BM25 and cosine scores live on different scales, so adding them lets BM25
    dominate. RRF only uses each document's rank, which keeps both searches
    on equal footing.
    """
    fused = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            entry = fused.setdefault(hit["_id"], {"hit": hit, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank + 1)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    results = []
    for entry in ranked[:top_k]:
        hit = dict(entry["hit"])
        hit["_score"] = entry["score"]
        results.append(hit)
    return results


def hybrid_search(query_text, top_k=20):
    """
    Run keyword and semantic search, then merge them with Reciprocal Rank Fusion.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search hits, with _score set to the fused RRF score
    """
    keyword_hits = keyword_search(query_text, top_k=top_k)
    semantic_hits = semantic_search(query_text, top_k=top_k)
    return reciprocal_rank_fusion([keyword_hits, semantic_hits], top_k)


@lru_cache(maxsize=1)
def _get_reranker():
    """Load the cross-encoder once. Returns None if it is unavailable."""
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s", config.RERANKER_MODEL)
        return CrossEncoder(config.RERANKER_MODEL)
    except Exception:
        logger.exception("Reranker unavailable; results will not be reranked")
        return None


def rerank(query_text, hits, top_k):
    """
    Reorder hits with a cross-encoder, which reads the query and each chunk
    together and scores relevance far more precisely than the first-stage search.
    """
    reranker = _get_reranker()
    if reranker is None or not hits:
        return hits[:top_k]

    pairs = [(query_text, hit["_source"].get("content", "")) for hit in hits]
    scores = reranker.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]


def search(query_text, search_type="hybrid", top_k=None, use_reranker=True):
    """
    Retrieve the most relevant chunks for a query.

    Fetches config.CANDIDATES results with the chosen search method, then
    optionally reranks them and keeps the best top_k.
    """
    top_k = top_k or config.TOP_K
    candidates = max(config.CANDIDATES, top_k)

    if search_type == "keyword":
        hits = keyword_search(query_text, top_k=candidates)
    elif search_type == "semantic":
        hits = semantic_search(query_text, top_k=candidates)
    else:
        hits = hybrid_search(query_text, top_k=candidates)

    if use_reranker:
        return rerank(query_text, hits, top_k)
    return hits[:top_k]


if __name__ == "__main__":
    for hit in search("Compare RAG v/s fine-tuning", "hybrid", top_k=5):
        source = hit["_source"]
        score = hit.get("rerank_score", hit["_score"])
        print(f"{score:.4f} [{source['content_type']}] {source['content'][:100]!r}")
