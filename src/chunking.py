from pydantic import BaseModel
from .models import Chunk
import ast

def split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    for i in range(0, len(lines) - 1):
        lines[i] = lines[i] + "\n"
    return lines

def split_line_by_max(text: str, max_len: int) -> list[str]:
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

def chunk_by_lines(lines: list[str], path: str, start_index: int, max_len: int = 2000) -> tuple[int, list[Chunk]]:
    chunks = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > max_len:
            if current_len > 0:
                chunks.append(Chunk(file_path = path, first_character_index = start_index, last_character_index = start_index + current_len))
                start_index += current_len
            if len(line) > max_len:
                sub_line = split_line_by_max(line, max_len)
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
    return start_index + current_len, chunks

def python_chunking(text:str, path: str, max_len: int = 2000) -> list[Chunk]:
    try:
        node_tree = ast.parse(text).body
    except (SyntaxError, ValueError):
        node_tree = []
    chunks = []
    start = 0
    start_index = 0
    lines = split_lines(text)
    for node in node_tree:
            start_index, temp_chunks = chunk_by_lines(lines[start:node.end_lineno], path, start_index, max_len)
            chunks.extend(temp_chunks)
            start = node.end_lineno
    if start < len(lines):
        start_index, temp_chunks = chunk_by_lines(lines[start:], path, start_index, max_len)
        chunks.extend(temp_chunks)
    return chunks