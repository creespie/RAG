"""Retrieval evaluation: IoU, recall@k and failure diagnosis."""

import sys
from collections import Counter

from .models import (AnsweredQuestion, MinimalSearchResults, MinimalSource,
                     RagDataset, StudentSearchResults)

IOU_THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Return the Intersection over Union of two character ranges.

    Two sources in different files have an IoU of 0.

    Args:
        a: First source.
        b: Second source.

    Returns:
        A value between 0 and 1.
    """
    if a.file_path != b.file_path:
        return 0.0
    inter = max(0, min(a.last_character_index, b.last_character_index)
                - max(a.first_character_index, b.first_character_index))
    union = ((a.last_character_index - a.first_character_index)
             + (b.last_character_index - b.first_character_index)
             - inter)
    return inter / union if union > 0 else 0.0


def best_iou(expected: MinimalSource, found: list[MinimalSource]) -> float:
    """Return the best IoU between one expected source and a result list."""
    return max((iou(expected, f) for f in found), default=0.0)


def question_recall(found: list[MinimalSource],
                    expected: list[MinimalSource], k: int) -> float:
    """Return the share of expected sources found in the first k results.

    Args:
        found: Retrieved sources, best first.
        expected: Reference sources of the question.
        k: Number of retrieved sources to consider.

    Returns:
        A value between 0 and 1.
    """
    if not expected:
        return 0.0
    top = found[:k]
    hits = sum(1 for e in expected if best_iou(e, top) >= IOU_THRESHOLD)
    return hits / len(expected)


def pair_questions(results: StudentSearchResults, dataset: RagDataset
                   ) -> list[tuple[AnsweredQuestion, MinimalSearchResults]]:
    """Match each answered question with its search result, by id."""
    by_id = {r.question_id: r for r in results.search_results}
    pairs = []
    for q in dataset.rag_questions:
        if isinstance(q, AnsweredQuestion) and q.sources:
            result = by_id.get(q.question_id)
            if result is not None:
                pairs.append((q, result))
    return pairs


def recall_at_k(pairs: list[tuple[AnsweredQuestion, MinimalSearchResults]],
                k: int) -> float:
    """Return the average recall@k over all the paired questions."""
    if not pairs:
        return 0.0
    return sum(question_recall(r.retrieved_sources, q.sources, k)
               for q, r in pairs) / len(pairs)


def load_pairs(results_path: str, dataset_path: str
               ) -> list[tuple[AnsweredQuestion, MinimalSearchResults]]:
    """Read a results file and a ground-truth dataset and pair them."""
    with open(results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate_json(f.read())
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    return pair_questions(results, dataset)


def span(s: MinimalSource) -> str:
    """Format a source as 'path [start:end] (length)'."""
    length = s.last_character_index - s.first_character_index
    return (f"{s.file_path} [{s.first_character_index}:"
            f"{s.last_character_index}] ({length} chars)")


def diagnose(results_path: str, dataset_path: str, k: int = 5,
             show: int = 15) -> None:
    """Print recall, general statistics and the failed questions.

    Args:
        results_path: Output of search_dataset.
        dataset_path: AnsweredQuestions dataset with the reference sources.
        k: Cut-off used to decide which questions failed.
        show: Maximum number of failed questions to print in detail.
    """
    pairs = load_pairs(results_path, dataset_path)
    print(f"Questions evaluated: {len(pairs)}")
    for cut in (1, 3, 5, 10):
        print(f"Recall@{cut}: {recall_at_k(pairs, cut):.3f}")

    expected_lengths = [s.last_character_index - s.first_character_index
                        for q, _ in pairs for s in q.sources]
    found_lengths = [s.last_character_index - s.first_character_index
                     for _, r in pairs for s in r.retrieved_sources[:k]]
    extensions = Counter(s.file_path.rsplit(".", 1)[-1]
                         for _, r in pairs for s in r.retrieved_sources[:k])
    print("\n--- Statistics ---")
    print(f"Expected sources: {len(expected_lengths)}, average length "
          f"{sum(expected_lengths) / max(1, len(expected_lengths)):.0f}, "
          f"min {min(expected_lengths, default=0)}, "
          f"max {max(expected_lengths, default=0)}")
    print(f"My top-{k} results: average length "
          f"{sum(found_lengths) / max(1, len(found_lengths)):.0f}, "
          f"by extension {dict(extensions)}")

    # Classify every expected source that was missed in the top-k.
    reasons: Counter[str] = Counter()
    failed = []
    for q, r in pairs:
        top = r.retrieved_sources[:k]
        missed = [s for s in q.sources if best_iou(s, top) < IOU_THRESHOLD]
        if not missed:
            continue
        failed.append((q, r, missed))
        for s in missed:
            top_files = {f.file_path for f in top}
            all_files = {f.file_path for f in r.retrieved_sources}
            if s.file_path in top_files:
                reasons["right file in top-k, IoU too low"] += 1
            elif best_iou(s, r.retrieved_sources) >= IOU_THRESHOLD:
                reasons["right chunk found, but after position k"] += 1
            elif s.file_path in all_files:
                reasons["right file after position k, wrong span"] += 1
            else:
                reasons["right file never retrieved"] += 1

    print(f"\n--- Missed sources at k={k}, by reason ---")
    for reason, count in reasons.most_common():
        print(f"{count:4d}  {reason}")

    print(f"\n--- First {min(show, len(failed))} failed questions "
          f"of {len(failed)} ---")
    for q, r, missed in failed[:show]:
        print("\n" + "=" * 70)
        print(f"Q: {q.question}")
        for s in missed:
            print(f"  expected: {span(s)}")
        for pos, f in enumerate(r.retrieved_sources[:k]):
            score = max(iou(s, f) for s in missed)
            print(f"  #{pos + 1} IoU={score:.3f}  {span(f)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.evaluation <results.json> "
              "<answered_dataset.json> [k] [show]")
        sys.exit(1)
    try:
        diagnose(sys.argv[1], sys.argv[2],
                 int(sys.argv[3]) if len(sys.argv) > 3 else 5,
                 int(sys.argv[4]) if len(sys.argv) > 4 else 15)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")