"""Split files into chunks: Python, Markdown and line-based strategies."""

import ast

from .models import Chunk


def split_lines(text: str) -> list[str]:
    """Split a text into lines, each one keeping its trailing newline.

    Args:
        text: Full content of a file.

    Returns:
        The lines; joined together they give back the original text.
    """
    lines = text.split("\n")
    for i in range(0, len(lines) - 1):
        lines[i] = lines[i] + "\n"
    return lines


def split_line_by_max(text: str, max_len: int) -> list[str]:
    """Cut a text into pieces of at most max_len characters.

    Args:
        text: Text to cut, usually a single line.
        max_len: Maximum length of a piece.

    Returns:
        The pieces, in order.
    """
    sections: list[str] = []
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


def chunk_by_lines(lines: list[str], path: str, start_index: int,
                   max_len: int = 2000) -> tuple[int, list[Chunk]]:
    """Group consecutive lines into chunks of at most max_len characters.

    A line longer than max_len is cut into pieces of max_len characters.

    Args:
        lines: Lines to group, each with its trailing newline.
        path: File path stored in each chunk.
        start_index: Character offset of the first line in the file.
        max_len: Maximum length of a chunk.

    Returns:
        The offset right after the last line, and the chunks.
    """
    chunks: list[Chunk] = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > max_len:
            if current_len > 0:
                chunks.append(Chunk(
                    file_path=path,
                    first_character_index=start_index,
                    last_character_index=start_index + current_len,
                ))
                start_index += current_len
            if len(line) > max_len:
                sub_line = split_line_by_max(line, max_len)
                for sub in range(len(sub_line) - 1):
                    piece_len = len(sub_line[sub])
                    chunks.append(Chunk(
                        file_path=path,
                        first_character_index=start_index,
                        last_character_index=start_index + piece_len,
                    ))
                    start_index += piece_len
                current_len = len(sub_line[-1])
            else:
                current_len = len(line)
        else:
            current_len += len(line)
    if current_len > 0:
        chunks.append(Chunk(
            file_path=path,
            first_character_index=start_index,
            last_character_index=start_index + current_len,
        ))
    return start_index + current_len, chunks


def python_chunking(text: str, path: str,
                    max_len: int = 2000) -> list[Chunk]:
    """Split a Python file into chunks following its top-level nodes.

    Each chunk goes from the end of the previous node to the end of the
    current one, so decorators and comments stay with the node that follows
    them. If the file cannot be parsed, it is split by lines.

    Args:
        text: Full content of the file.
        path: File path stored in each chunk.
        max_len: Maximum length of a chunk.

    Returns:
        The chunks, covering the whole file.
    """
    try:
        node_tree = ast.parse(text).body
    except (SyntaxError, ValueError):
        node_tree = []
    chunks: list[Chunk] = []
    start = 0
    start_index = 0
    lines = split_lines(text)
    for node in node_tree:
        end = node.end_lineno or node.lineno
        start_index, temp_chunks = chunk_by_lines(
            lines[start:end], path, start_index, max_len)
        chunks.extend(temp_chunks)
        start = end
    if start < len(lines):
        start_index, temp_chunks = chunk_by_lines(
            lines[start:], path, start_index, max_len)
        chunks.extend(temp_chunks)
    return chunks


def is_title(text: str, level: int = 2) -> bool:
    """Tell whether a line is a Markdown heading of at most a given level.

    Args:
        text: The line.
        level: Maximum number of '#' characters accepted.

    Returns:
        True for at most 3 spaces, 1 to level '#', then a space or a tab.
    """
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
    """Return the line numbers where a new Markdown section starts.

    Headings inside fenced code blocks are ignored, and a heading that is
    not preceded by any content does not start a new section.

    Args:
        lines: Lines of the file.
        level: Maximum heading level that starts a section.

    Returns:
        The indexes of the lines where the file must be cut.
    """
    ends: list[int] = []
    text = False
    comment_tide = False
    comment_ap = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        in_block = comment_tide or comment_ap
        if stripped.startswith("```") and not in_block:
            comment_ap = True
            text = True
        elif stripped.startswith("~~~") and not in_block:
            comment_tide = True
            text = True
        elif comment_tide:
            if stripped == "~~~":
                comment_tide = False
        elif comment_ap:
            if stripped == "```":
                comment_ap = False
        elif is_title(line, level):
            if text:
                text = False
                ends.append(i)
        elif stripped and not text:
            text = True
    return ends


def md_chunking(text: str, path: str, max_len: int,
                level: int = 2) -> list[Chunk]:
    """Split a Markdown file into chunks, cutting before its headings.

    Args:
        text: Full content of the file.
        path: File path stored in each chunk.
        max_len: Maximum length of a chunk.
        level: Maximum heading level that starts a new chunk.

    Returns:
        The chunks, covering the whole file.
    """
    lines = split_lines(text)
    title_tree = split_titles(lines, level)
    chunks: list[Chunk] = []
    start = 0
    start_index = 0
    for node in title_tree:
        start_index, temp_chunks = chunk_by_lines(
            lines[start:node], path, start_index, max_len)
        chunks.extend(temp_chunks)
        start = node
    if start < len(lines):
        start_index, temp_chunks = chunk_by_lines(
            lines[start:], path, start_index, max_len)
        chunks.extend(temp_chunks)
    return chunks
