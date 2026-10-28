from pathlib import Path

def file_list(location: str = "data/raw/") -> list[str]:
    path = Path(location)
    valid_files = []
    for file in path.rglob("*"):
        if any(section.startswith(".") for section in file.relative_to(location).parts):
            continue
        if file.suffix.lower() in (".py", ".md") and file.is_file():
            valid_files.append(file.as_posix())

    valid_files.sort()
    return valid_files

def read_file(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
            
        return text
    except UnicodeDecodeError:
        print(f"{path} is not a valid text file and will be skipped")
    except OSError as error:
        print(f"{path} is impossible to read: {error}")

    return None
    