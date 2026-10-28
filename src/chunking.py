from pydantic import BaseModel
from .models import Chunk

def split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    for i in range(0, len(lines) - 1):
        lines[i] = lines[i] + "\n"
    return lines

def split_by_max(text: str, max_len: int) -> list[str]:
    sections = []
    total_len = len(text)
    current_i = 0
    while current_i < total_len:
        if total_len - current_i > max_len:
            sections.append(text[current_i:current_i + max_len])
            current_i += max_len
        else:
            sections.append(text[current_i:])
            current_i = total_len
    return sections

def chunk_by_lines(text: str, path: str, start_index: int, max_len: int = 2000) -> list[Chunk]:
    lines = split_lines(text)
    chunks = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > max_len:
            if current_len > 0:
                chunks.append(Chunk(file_path = path, first_character_index = start_index, last_character_index = start_index + current_len))
                start_index += current_len
            if len(line) > max_len:
                sub_line = split_by_max(line, max_len)
                for sub in range(len(sub_line) - 1):
                    chunks.append(Chunk(file_path = path, first_character_index = start_index, last_character_index = start_index + len(sub_line[sub])))
                    start_index += len(sub_line[sub])
                current_len = len(sub_line[-1])
            else:
                current_len = len(line)
        else:
            current_len += len(line)
    if current_len > 0:
        chunks.append(Chunk(file_path = path, first_character_index = start_index, last_character_index = start_index + current_len))
    return chunks
