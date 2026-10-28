"""Tests for the line-based and Python chunking functions."""

from src.chunking import chunk_by_lines, python_chunking, split_lines
from src.models import Chunk

PATH = "data/raw/test/file.py"

EXAMPLE = '''import os

X = 5

@decoratore
def f(a):
    return a

class A:
    """Una classe."""
    y = 1

    def m(self):
        pass
'''


def texts(text: str, chunks: list[Chunk]) -> list[str]:
    """Return the text covered by each chunk."""
    return [text[c.first_character_index:c.last_character_index]
            for c in chunks]


def check_coverage(text: str, chunks: list[Chunk], max_len: int) -> None:
    """Check the invariants every chunking result must respect.

    The chunks must cover the whole text, in order, with no gaps and no
    overlaps, none may be empty, none may exceed max_len, and every one
    must carry the expected file path.
    """
    if text == "":
        assert chunks == []
        return
    assert chunks, "a non-empty text must produce at least one chunk"
    assert chunks[0].first_character_index == 0
    assert chunks[-1].last_character_index == len(text)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.first_character_index == \
            previous.last_character_index, "gap or overlap between chunks"
    for c in chunks:
        length = c.last_character_index - c.first_character_index
        assert 0 < length <= max_len, f"bad chunk length {length}"
        assert c.file_path == PATH
    assert "".join(texts(text, chunks)) == text


# --- split_lines ---------------------------------------------------------

def test_split_lines_keeps_newlines() -> None:
    assert split_lines("aaa\nbbbb\ncc") == ["aaa\n", "bbbb\n", "cc"]


def test_split_lines_rebuilds_text() -> None:
    for text in ["", "x", "a\n", "a\n\nb\n", EXAMPLE]:
        assert "".join(split_lines(text)) == text


# --- chunk_by_lines ------------------------------------------------------

def test_chunk_by_lines_groups_lines() -> None:
    text = "aaa\nbbbb\ncc"
    end, chunks = chunk_by_lines(split_lines(text), PATH, 0, 5)
    spans = [(c.first_character_index, c.last_character_index)
             for c in chunks]
    assert spans == [(0, 4), (4, 9), (9, 11)]
    assert end == 11


def test_chunk_by_lines_splits_long_line() -> None:
    text = "abcdefghijkl"
    _, chunks = chunk_by_lines(split_lines(text), PATH, 0, 5)
    spans = [(c.first_character_index, c.last_character_index)
             for c in chunks]
    assert spans == [(0, 5), (5, 10), (10, 12)]


def test_chunk_by_lines_respects_start_index() -> None:
    _, chunks = chunk_by_lines(["abc\n"], PATH, 100, 2000)
    assert chunks[0].first_character_index == 100
    assert chunks[0].last_character_index == 104


def test_chunk_by_lines_empty_input() -> None:
    end, chunks = chunk_by_lines([], PATH, 7, 2000)
    assert chunks == []
    assert end == 7


# --- python_chunking -----------------------------------------------------

def test_python_example_one_chunk_per_node() -> None:
    chunks = python_chunking(EXAMPLE, PATH)
    check_coverage(EXAMPLE, chunks, 2000)
    assert texts(EXAMPLE, chunks) == [
        "import os\n",
        "\nX = 5\n",
        "\n@decoratore\ndef f(a):\n    return a\n",
        '\nclass A:\n    """Una classe."""\n    y = 1\n\n'
        "    def m(self):\n        pass\n",
    ]


def test_python_decorator_stays_with_function() -> None:
    chunks = python_chunking(EXAMPLE, PATH)
    function_chunk = [t for t in texts(EXAMPLE, chunks) if "def f" in t]
    assert len(function_chunk) == 1
    assert "@decoratore" in function_chunk[0]


def test_python_small_max_len() -> None:
    for max_len in [1, 5, 20, 50]:
        check_coverage(EXAMPLE, python_chunking(EXAMPLE, PATH, max_len),
                       max_len)


def test_python_syntax_error_falls_back() -> None:
    text = "def f(:\n    pass\n"
    check_coverage(text, python_chunking(text, PATH), 2000)


def test_python_null_byte_does_not_crash() -> None:
    text = "x = 1\n\x00\n"
    check_coverage(text, python_chunking(text, PATH), 2000)


def test_python_only_comments() -> None:
    text = "# licenza\n# altro commento\n"
    chunks = python_chunking(text, PATH)
    check_coverage(text, chunks, 2000)
    assert len(chunks) == 1


def test_python_empty_file() -> None:
    assert python_chunking("", PATH) == []


def test_python_no_trailing_newline() -> None:
    text = "def f():\n    return 1"
    check_coverage(text, python_chunking(text, PATH), 2000)


def test_python_trailing_comments_are_kept() -> None:
    text = "def f():\n    return 1\n\n# commento finale\n"
    chunks = python_chunking(text, PATH)
    check_coverage(text, chunks, 2000)
    assert texts(text, chunks)[-1].endswith("# commento finale\n")


def test_python_long_single_line() -> None:
    text = "x = '" + "a" * 5000 + "'\n"
    check_coverage(text, python_chunking(text, PATH, 2000), 2000)