from .models import Chunk
from .corpus import file_list, read_file
from .chunking import md_chunking, python_chunking
from .tokenizer import chunk_splitter
from tqdm import tqdm
import time

def chunks_to_words(chnks, complete_test):
    words_list = []
    for chunk in chnks:
        words_list.append(chunk_splitter(complete_test[chunk.first_character_index:chunk.last_character_index]))
    return words_list


def indexing(source: str, max_len: int) -> list[Chunk]:
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


if __name__ == "__main__":
    import random

    max_len = 2000
    start = time.time()
    chunks, words = indexing("data/raw", max_len)
    elapsed = time.time() - start

    print(f"Tempo: {elapsed:.1f} s")
    print(f"Chunk: {len(chunks)}  Liste di parole: {len(words)}")
    assert len(chunks) == len(words), "le due liste non sono parallele!"

    py = sum(1 for c in chunks if c.file_path.endswith(".py"))
    print(f"Da .py: {py}  Da .md: {len(chunks) - py}")

    longest = max(c.last_character_index - c.first_character_index
                  for c in chunks)
    print(f"Chunk più lungo: {longest} caratteri (limite {max_len})")
    assert longest <= max_len, "c'è un chunk oltre il limite!"

    empty = sum(1 for w in words if not w)