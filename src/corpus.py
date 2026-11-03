"""Find and read the files of the corpus."""

from pathlib import Path

EXTENSIONS = (".py", ".md", ".txt")


def file_list(location: str = "data/raw/") -> list[str]:
    """List the files to index under a folder, in sorted order.

    Hidden folders (whose name starts with a dot) are skipped.

    Args:
        location: Root folder of the corpus.

    Returns:
        The paths of the .py, .md and .txt files, relative to the project
        root and written with '/'.
    """
    path = Path(location)
    valid_files: list[str] = []
    for file in path.rglob("*"):
        parts = file.relative_to(location).parts
        if any(section.startswith(".") for section in parts):
            continue
        if file.suffix.lower() in EXTENSIONS and file.is_file():
            valid_files.append(file.as_posix())
    valid_files.sort()
    return valid_files


def read_file(path: str) -> str | None:
    """Read a whole file as UTF-8 text.

    Args:
        path: Path of the file.

    Returns:
        The content of the file, or None if it cannot be read.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"{path} is not a valid text file and will be skipped")
    except OSError as error:
        print(f"{path} is impossible to read: {error}")
    return None
