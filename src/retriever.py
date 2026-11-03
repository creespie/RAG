"""Search the BM25 index for the chunks most relevant to a question."""

import bm25s

from .models import Chunk
from .tokenizer import STOPWORDS, chunk_splitter


def query_search(query: str, k: int, retriever: bm25s.BM25,
                 chunks: list[Chunk]) -> list[Chunk]:
    """Return the k chunks most relevant to a question, best first.

    Args:
        query: The question.
        k: Number of results wanted; reduced to the number of chunks.
        retriever: The BM25 index.
        chunks: The chunks, in the order used to build the index.

    Returns:
        The best chunks, or an empty list if the question has no useful
        word or no word known to the index.

    Raises:
        ValueError: If k is zero or negative.
    """
    if k <= 0:
        raise ValueError(
            f"In {query} the amount of answers is negative or zero")
    if k > len(chunks):
        k = len(chunks)

    query_words = [word for word in chunk_splitter(query)
                   if word not in STOPWORDS]
    if len(query_words) == 0:
        print(f"No relevant words in {query}")
        return []
    indices, scores = retriever.retrieve([query_words], k=k)
    if scores[0][0] == 0:
        print(f"No matching words in {query}")
        return []
    return [chunks[int(chunk)] for chunk in indices[0]]
