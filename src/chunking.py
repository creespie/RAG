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

def is_title(text: str, level: int = 2) -> bool:
    i = 0
    spaces = 0
    lev = 0
    while i < len(text):
        if lev == 0 and text[i] == " ":
            if spaces > 2:
                return False
            spaces += 1
            i += 1
        elif text[i] == "#":
            if lev >= level:
                return False
            lev += 1
            i += 1
        elif lev > 0 and text[i] in (" ", "\t"):
            return True
        else:
            return False
    return False


def split_titles(lines: list[str], level: int = 2) -> list[int]:
    ends = []
    text = False
    comment_tide = False
    comment_ap = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```") and not (comment_tide or comment_ap):
            comment_ap = True
            text = True
        elif line.strip().startswith("~~~") and not (comment_tide or comment_ap):
            comment_tide = True
            text = True
        elif comment_tide:
            if line.strip() == "~~~":
                comment_tide = False
        elif comment_ap:
            if line.strip() == "```":
                comment_ap = False
        elif is_title(line, level):
            if text:
                text = False
                ends.append(i)
        elif line.strip() and not text:
            text = True

    return ends



def md_chunking(text: str, path: str, max_len: int, level: int = 2):
    lines = split_lines(text)
    title_tree = split_titles(lines, level)
    chunks = []
    start = 0
    start_index = 0
    for node in title_tree:
            start_index, temp_chunks = chunk_by_lines(lines[start:node], path, start_index, max_len)
            chunks.extend(temp_chunks)
            start = node
    if start < len(lines):
        start_index, temp_chunks = chunk_by_lines(lines[start:], path, start_index, max_len)
        chunks.extend(temp_chunks)
    return chunks

