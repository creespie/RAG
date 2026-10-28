from .models import Chunk
from .corpus import file_list, read_file
from .chunking import md_chunking, python_chunking
from tqdm import tqdm
import time


def indexing(source: str, max_len: int) -> list[Chunk]:
    files = file_list(source)
    if not files:
        raise ValueError(f"No file found from {source}")
    chunks = []
    for file in tqdm(files, desc="Chunking", unit="file"):
        current = read_file(file)
        if current is None:
            continue
        if file.lower().endswith(".py"):
            chunks.extend(python_chunking(current, file, max_len))
        else:
            chunks.extend(md_chunking(current, file, max_len))
    return chunks
