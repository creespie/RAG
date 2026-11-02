import bisect

from .chunking import is_title, split_lines
from .models import Chunk


def md_headings(text: str) -> list[tuple[int, int, str]]:
    """Return every Markdown heading as (offset, level, title).

    Headings inside fenced code blocks are ignored.

    Args:
        text: Full content of a Markdown file.

    Returns:
        The headings in file order: character offset of the line,
        number of '#' characters and heading text without the '#'.
    """
    headings: list[tuple[int, int, str]] = []
    fence = ""
    offset = 0
    for line in split_lines(text):
        stripped = line.strip()
        if fence:
            if stripped == fence:
                fence = ""
        elif stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
        elif is_title(line, 6):
            title = stripped.lstrip("#")
            level = len(stripped) - len(title)
            headings.append((offset, level, title.strip()))
        offset += len(line)
    return headings


def heading_context(text: str, chunks: list[Chunk]) -> list[str]:
    """Return, for each chunk, the titles of the sections containing it.

    For a chunk, the result holds the page title (first level-1 heading)
    and the last heading of each level that appears before the chunk
    starts, e.g. "Using LoRA Serving LoRA adapters".

    Args:
        text: Full content of the Markdown file.
        chunks: Chunks of that file.

    Returns:
        One string per chunk, in the same order.
    """
    headings = md_headings(text)
    offsets = [h[0] for h in headings]
    contexts = []
    for chunk in chunks:
        # Headings that start at or before the beginning of the chunk.
        last = bisect.bisect_right(offsets, chunk.first_character_index)
        current: dict[int, str] = {}
        for _, level, title in headings[:last]:
            current[level] = title
            for deeper in [lv for lv in current if lv > level]:
                del current[deeper]
        contexts.append(" ".join(current[lv] for lv in sorted(current)))
    return contexts
