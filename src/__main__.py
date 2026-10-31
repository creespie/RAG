from .indexer import run_index, load_index
from .retriever import query_search
import fire


class CLI:

    def index(self, max_chunk_size: int = 2000, input_path: str = "data/raw", output_path: str = "data/processed") -> None:
        try:
            if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
                raise ValueError("max_chunk_size must be a positive number greater than 0")
            run_index(input_path, output_path, max_chunk_size)
        except Exception as e:
            print(e)

    def search(self, query: str, k: int = 10, index_folder: str = "data/processed"):
        try:
            query = str(query)
            if not isinstance(k, int) or k <= 0 or not isinstance(k, bool):
                raise ValueError("k must be a positive number greater than 0")
            chunks, bm = load_index(index_folder)
            found = query_search(query, k, bm, chunks)
            for istance in found:
                print(f"{istance.file_path} [{istance.first_character_index}: {istance.last_character_index}]")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    fire.Fire(CLI)