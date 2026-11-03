"""Command-line interface of the RAG system."""

from pathlib import Path

import fire
from pydantic import ValidationError
from tqdm import tqdm

from .evaluation import load_pairs, recall_at_k
from .generator import NO_CONTEXT_ANSWER, generate_answer, load_model
from .indexer import load_index, run_index
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
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

    def index(
        self,
        max_chunk_size: int = 2000,
        input_path: str = "data/raw",
        output_path: str = "data/processed",
    ) -> None:
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

    def search(
        self, query: str, k: int = 10, index_folder: str = "data/processed"
    ) -> None:
        """Print the top-k sources for a single query."""
        try:
            k = check_k(k)
            chunks, bm = load_index(index_folder)
            for source in query_search(str(query), k, bm, chunks):
                print(
                    f"{source.file_path} [{source.first_character_index}:"
                    f"{source.last_character_index}]"
                )
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
        index_folder: str = "data/processed",
    ) -> None:
        """Search every question of a dataset and save the results."""
        try:
            k = check_k(k)

            # 1. Read the questions.
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())

            # 2. Load the index once, for all the questions.
            chunks, bm = load_index(index_folder)

            # 3. Search each question and collect its results.
            results: list[MinimalSearchResults] = []
            for question in tqdm(
                dataset.rag_questions, desc="Searching", unit="question"
            ):
                sources: list[MinimalSource] = list(
                    query_search(question.question, k, bm, chunks))
                results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources,
                    )
                )

            # 4. Write the output, with the same file name as the dataset.
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

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall@k of a search results file against a dataset."""
        try:
            pairs = load_pairs(student_search_results_path, dataset_path)
            if not pairs:
                raise ValueError(
                    "No question of the dataset has both "
                    "reference sources and search results"
                )
            print(f"Questions evaluated: {len(pairs)}")
            for k in (1, 3, 5, 10):
                print(f"Recall@{k}: {recall_at_k(pairs, k):.3f}")
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")

    def answer(
        self,
        query: str,
        k: int = 5,
        index_folder: str = "data/processed",
        max_context_tokens: int = 2000,
    ) -> None:
        """Answer a single query using the retrieved sources."""
        try:
            k = check_k(k)
            max_context_tokens = check_k(max_context_tokens)
            chunks, bm = load_index(index_folder)
            sources = query_search(str(query), k, bm, chunks)
            if sources:
                tokenizer, model = load_model()
                text = generate_answer(
                    str(query),
                    list(sources),
                    tokenizer,
                    model,
                    max_context_tokens=max_context_tokens,
                )
            else:
                text = NO_CONTEXT_ANSWER
            print(f"\nAnswer:\n{text}\n\nSources:")
            for source in sources:
                print(
                    f"  {source.file_path} [{source.first_character_index}:"
                    f"{source.last_character_index}]"
                )
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        max_context_tokens: int = 2000,
    ) -> None:
        """Answer every question of a search results file and save them."""
        try:
            max_context_tokens = check_k(max_context_tokens)
            with open(student_search_results_path, encoding="utf-8") as f:
                results = StudentSearchResults.model_validate_json(f.read())
            total = len(results.search_results)
            print(f"Loaded {total} questions")

            tokenizer, model = load_model()
            cache: dict[str, str] = {}
            answers: list[MinimalAnswer] = []
            for result in tqdm(
                results.search_results, desc="Answering", unit="question"
            ):
                text = generate_answer(
                    result.question,
                    result.retrieved_sources,
                    tokenizer,
                    model,
                    cache,
                    max_context_tokens,
                )
                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question=result.question,
                        retrieved_sources=result.retrieved_sources,
                        answer=text,
                    )
                )
            print(f"Processed {len(answers)} of {total} questions")

            output = StudentSearchResultsAndAnswer(search_results=answers,
                                                   k=results.k)
            folder = Path(save_directory)
            folder.mkdir(parents=True, exist_ok=True)
            output_file = folder / Path(student_search_results_path).name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
            print(f"Saved student_search_results_and_answer to {output_file}")
        except (ValueError, OSError, ValidationError) as error:
            print(f"Error: {error}")
        except Exception as error:
            print(f"Unexpected error: {error}")


if __name__ == "__main__":
    fire.Fire(CLI)
