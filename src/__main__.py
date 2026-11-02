"""Command-line interface of the RAG system."""

from pathlib import Path

import fire
from pydantic import ValidationError
from tqdm import tqdm

from .indexer import load_index, run_index
from .models import MinimalSearchResults, RagDataset, StudentSearchResults
from .retriever import query_search


def check_k(k: object) -> int:
    """Check that k is a positive integer and return it.

    Args:
        k: Value received from the command line.

    Returns:
        The same value, typed as int.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return k


class CLI:
    """RAG system over the vLLM codebase."""

    def index(self, max_chunk_size: int = 2000,
              input_path: str = "data/raw",
              output_path: str = "data/processed") -> None:
        """Chunk the corpus, build the BM25 index and save it."""
        try:
            if (isinstance(max_chunk_size, bool)
                    or not isinstance(max_chunk_size, int)):
                raise ValueError("max_chunk_size must be an integer")
            run_index(input_path, output_path, max_chunk_size)
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")

    def search(self, query: str, k: int = 10,
               index_folder: str = "data/processed") -> None:
        """Print the top-k sources for a single query."""
        try:
            k = check_k(k)
            chunks, bm = load_index(index_folder)
            for source in query_search(str(query), k, bm, chunks):
                print(f"{source.file_path} [{source.first_character_index}:"
                      f"{source.last_character_index}]")
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10,
                       index_folder: str = "data/processed") -> None:
        """Search every question of a dataset and save the results."""
        try:
            k = check_k(k)
    
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())

            chunks, bm = load_index(index_folder)

            results: list[MinimalSearchResults] = []
            for question in tqdm(dataset.rag_questions, desc="Searching",
                                 unit="question"):
                sources = query_search(question.question, k, bm, chunks)
                results.append(MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                ))

            output = StudentSearchResults(search_results=results, k=k)
            folder = Path(save_directory)
            folder.mkdir(parents=True, exist_ok=True)
            output_file = folder / Path(dataset_path).name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
            print(f"Saved student_search_results to {output_file}")
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")


if __name__ == "__main__":
    fire.Fire(CLI)