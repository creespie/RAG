from .models import Chunk, IndexData
from .corpus import file_list, read_file
from .chunking import md_chunking, python_chunking
from .tokenizer import chunk_splitter
from tqdm import tqdm
import time
import bm25s
from pathlib import Path
from pydantic import ValidationError


def chunks_to_words(chnks, complete_test):
    words_list = []
    for chunk in chnks:
        words_list.append(chunk_splitter(complete_test[chunk.first_character_index:chunk.last_character_index]))
    return words_list


def build_chunks(source: str, max_len: int) -> list[Chunk]:
    files = file_list(source)
    if not files:
        raise ValueError(f"No file found from {source}")
    chunks = []
    words = []
    for file in tqdm(files, desc="Chunking", unit="file"):
        current = read_file(file)
        if current is None:
            continue
        if file.lower().endswith(".py"):
            tmp = python_chunking(current, file, max_len)
            chunks.extend(tmp)
            words.extend(chunks_to_words(tmp, current))
        else:
            tmp = md_chunking(current, file, max_len)
            chunks.extend(tmp)
            words.extend(chunks_to_words(tmp, current))
    return chunks, words


def build_bm25_index(k_value, b_value, words):
    retriever = bm25s.BM25(k1=k_value, b=b_value)
    retriever.index(words)
    return retriever


def save_index(retriever, chunks, dest_path):
    folder = Path(dest_path)
    folder.mkdir(parents=True, exist_ok=True)

    file_chunks = folder / "chunks.json"
    folder_bm25 = folder / "bm25"
    retriever.save(folder_bm25)
    with open(file_chunks, "w", encoding="utf-8") as f:
        f.write(IndexData(chunks=chunks).model_dump_json())


def load_index(path):
    pth = Path(path)
    try:
        with open(pth / "chunks.json", encoding="utf-8") as f:
            chnks = f.read()
        chunks = IndexData.model_validate_json(chnks).chunks
        bm = bm25s.BM25.load(pth / "bm25")
        return chunks, bm
    except (OSError, ValidationError) as error:
        raise FileNotFoundError(f"Index not found in {path}, run the index command first: {error}")

def run_index(input_path, ouput_path, max_chunk_size, k1 = 1.2, b = 0.75):
    if max_chunk_size < 100:
        raise ValueError("The max chunk size can't be lower than 100")
    chunks, words = build_chunks(input_path, max_chunk_size)
    retriever = build_bm25_index(k1, b, words)
    save_index(retriever, chunks, ouput_path)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under {input_path}")


if __name__ == "__main__":
    import random

    run_index("data/raw", "data/processed", 2000)
    chunks, bm = load_index("data/processed")

    print(len(chunks))
    query = ["lora", "rank"]

    indices, scores = bm.retrieve([query], k=5)
    print(indices)
    print(scores)
    print(chunks[indices[0][0]])
