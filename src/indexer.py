"""Build the BM25 index of the corpus, save it and load it back."""

from pathlib import Path

import bm25s
from pydantic import ValidationError
from tqdm import tqdm

from .chunking import chunk_by_lines, md_chunking, python_chunking, split_lines
from .corpus import file_list, read_file
from .headings import heading_context
from .models import Chunk, IndexData
from .tokenizer import chunk_splitter

MIN_CHUNK_SIZE = 100


def chunks_to_words(chnks: list[Chunk], complete_text: str,
                    path: str) -> list[list[str]]:
    """Tokenize each chunk, adding the words of its path and headings.

    Args:
        chnks: Chunks of one file.
        complete_text: Full content of that file.
        path: Path of the file.

    Returns:
        One list of words per chunk, in the same order as the chunks.
    """
    if path.lower().endswith(".md"):
        contexts = heading_context(complete_text, chnks)
    else:
        contexts = [""] * len(chnks)
    path_words = chunk_splitter(path)
    words_list: list[list[str]] = []
    for chunk, context in zip(chnks, contexts):
        text = complete_text[chunk.first_character_index:
                             chunk.last_character_index]
        words_list.append(chunk_splitter(text) + path_words
                          + chunk_splitter(context))
    return words_list


def build_chunks(source: str,
                 max_len: int) -> tuple[list[Chunk], list[list[str]]]:
    """Chunk and tokenize every file of the corpus.

    Args:
        source: Root folder of the corpus.
        max_len: Maximum length of a chunk.

    Returns:
        The chunks and, in the same order, the words of each chunk.

    Raises:
        ValueError: If no file to index is found.
    """
    files = file_list(source)
    if not files:
        raise ValueError(f"No file found from {source}")
    chunks: list[Chunk] = []
    words: list[list[str]] = []
    for file in tqdm(files, desc="Chunking", unit="file"):
        current = read_file(file)
        if current is None:
            continue
        if file.lower().endswith(".py"):
            tmp = python_chunking(current, file, max_len)
        elif file.lower().endswith(".md"):
            tmp = md_chunking(current, file, max_len)
        else:
            _, tmp = chunk_by_lines(split_lines(current), file, 0, max_len)
        chunks.extend(tmp)
        words.extend(chunks_to_words(tmp, current, file))
    return chunks, words


def build_bm25_index(k_value: float, b_value: float,
                     words: list[list[str]]) -> bm25s.BM25:
    """Build a BM25 index from the words of each chunk.

    Args:
        k_value: BM25 k1 parameter (term frequency saturation).
        b_value: BM25 b parameter (length normalization).
        words: One list of words per chunk.

    Returns:
        The BM25 index.
    """
    retriever = bm25s.BM25(k1=k_value, b=b_value)
    retriever.index(words)
    return retriever


def save_index(retriever: bm25s.BM25, chunks: list[Chunk],
               dest_path: str) -> None:
    """Save the BM25 index and the chunks in a folder.

    Args:
        retriever: The BM25 index.
        chunks: The chunks, in the order used to build the index.
        dest_path: Destination folder, created if missing.
    """
    folder = Path(dest_path)
    folder.mkdir(parents=True, exist_ok=True)
    retriever.save(folder / "bm25")
    with open(folder / "chunks.json", "w", encoding="utf-8") as f:
        f.write(IndexData(chunks=chunks).model_dump_json())


def load_index(path: str) -> tuple[list[Chunk], bm25s.BM25]:
    """Load the chunks and the BM25 index saved by save_index.

    Args:
        path: Folder of the index.

    Returns:
        The chunks and the BM25 index.

    Raises:
        FileNotFoundError: If the index is missing or invalid.
    """
    pth = Path(path)
    try:
        with open(pth / "chunks.json", encoding="utf-8") as f:
            chunks = IndexData.model_validate_json(f.read()).chunks
        bm = bm25s.BM25.load(pth / "bm25")
        return chunks, bm
    except (OSError, ValidationError) as error:
        raise FileNotFoundError(
            f"Index not found in {path}, run the index command first: "
            f"{error}") from error


def run_index(input_path: str, output_path: str, max_chunk_size: int,
              k1: float = 1.2, b: float = 0.5) -> None:
    """Chunk the corpus, build the BM25 index and save it.

    Args:
        input_path: Root folder of the corpus.
        output_path: Folder where the index is saved.
        max_chunk_size: Maximum length of a chunk.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.

    Raises:
        ValueError: If max_chunk_size is too small or no file is found.
    """
    if max_chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(
            f"The max chunk size can't be lower than {MIN_CHUNK_SIZE}")
    chunks, words = build_chunks(input_path, max_chunk_size)
    retriever = build_bm25_index(k1, b, words)
    save_index(retriever, chunks, output_path)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks "
          f"under {output_path}")
