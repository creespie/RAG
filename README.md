*This project has been created as part of the 42 curriculum by lurossi.*

# RAG against the machine

> **Status:** work in progress. Indexing (chunking, tokenization, BM25 index,
> persistence) and single-query retrieval are implemented. The CLI, dataset
> search, evaluation and answer generation are still to be done. Sections
> marked *TODO* will be completed as the project progresses.

## Description

This project is a **Retrieval-Augmented Generation (RAG)** system that answers
questions about a codebase: the source code and documentation of
[vLLM](https://github.com/vllm-project/vllm) 0.10.1.

A language model only knows what it saw during training. Instead of retraining
it, a RAG system looks up the relevant information at question time and hands
it to the model together with the question. The pipeline has four stages:

1. **Indexing** – split every file of the corpus into chunks and build a
   searchable index.
2. **Retrieving** – given a question, find the most relevant chunks.
3. **Augmenting** – place those chunks in the model's context.
4. **Generating** – let a small local model (`Qwen/Qwen3-0.6B`) write an answer
   grounded in them.

The system is evaluated mainly on retrieval quality: for each question it must
return the right **source locations** (file path and character range), measured
with recall@k.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) as package and project manager

### Installation

```bash
make install        # runs: uv sync
```

The vLLM repository must be placed in `data/raw/` (e.g.
`data/raw/vllm-0.10.1/`) and the question datasets in `data/datasets/`.
The `data/` folder is not part of the repository.

### Makefile rules

| Rule    | Description                                      |
|---------|--------------------------------------------------|
| install | Install the dependencies with `uv sync`          |
| run     | Run the program (`uv run python -m src`)         |
| debug   | Run the program under `pdb`                      |
| clean   | Remove caches (`__pycache__`, `.mypy_cache`)     |
| lint    | Run `flake8` and `mypy` with the required flags  |

### Commands

*TODO: the CLI (Python Fire) is not implemented yet.* Planned commands, as
required by the subject:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "how to configure LoRA" --k 5
uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir>
uv run python -m src answer "how to configure LoRA" --k 5
uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>
uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>
```

## System architecture

```
data/raw/ ──► corpus.py ──► chunking.py ──► tokenizer.py ──► indexer.py ──► data/processed/
                (files)       (chunks)        (words)        (BM25 + save)
                                                                   │
question ──► tokenizer.py ──► retriever.py ◄── load_index ─────────┘
              (+ stopwords)    (top-k sources)
```

| Module         | Responsibility                                                     |
|----------------|--------------------------------------------------------------------|
| `corpus.py`    | Lists the `.py` and `.md` files under the corpus folder (skipping hidden folders) and reads them as UTF-8 text, skipping unreadable files. |
| `chunking.py`  | Splits a file into chunks: one strategy for Python, one for Markdown, and a line-based fallback. |
| `tokenizer.py` | Turns a text into a list of words, and holds the stopword list.    |
| `indexer.py`   | Builds the chunks and the BM25 index, saves them and loads them back. |
| `retriever.py` | Returns the top-k chunks for a question.                           |
| `models.py`    | Pydantic models: the ones required by the subject, plus `Chunk` and `IndexData`. |

Each module does one thing and receives its input as parameters: the chunkers
work on a string and never open files, so they can be tested on hand-written
inputs.

## Chunking strategy

A chunk is identified by a file path and a character range
(`first_character_index`, `last_character_index`, end excluded). The chunks of
a file are **contiguous and cover the whole file**: every character belongs to
exactly one chunk, so no part of the corpus is invisible to retrieval. No chunk
is ever longer than `max_chunk_size`.

### Line-based fallback

Used for oversized pieces and for files that cannot be parsed. Lines (each
keeping its trailing `\n`, so lengths add up exactly) are accumulated until the
next one would exceed the limit. A single line longer than the limit is split
into pieces of exactly `max_chunk_size` characters.

### Python files

The file is parsed with the standard `ast` module. Each chunk goes from the
end of the previous top-level node to the end of the current one. As a
consequence:

- decorators, blank lines and comments placed before a function or class are
  kept with it;
- trailing comments at the end of the file form a final chunk;
- oversized nodes are split with the line-based fallback.

If the file cannot be parsed (`SyntaxError`, or `ValueError` for null bytes),
the whole file is split by lines.

### Markdown files

The file is split **before** each heading of level 1 or 2 (`#` and `##`; the
level is configurable). A line is a heading if it has at most 3 leading spaces,
1 to `level` `#` characters, then a space or tab. Two details:

- **Fenced code blocks** (```` ``` ```` and `~~~`, also when indented inside
  MkDocs tabs) are tracked, so a shell or Python comment such as
  `# install vllm` is not taken for a heading. A block is only closed by the
  same kind of fence that opened it.
- **Consecutive headings without content** between them are kept together, so
  a page title is not left alone in a one-line chunk but stays attached to the
  first section.

Oversized sections are split with the line-based fallback.

### Minimum chunk size

`max_chunk_size` must be at least 100. With extremely small values (e.g. 1)
the number of chunks becomes equal to the number of characters of the corpus,
and memory usage explodes.

## Retrieval method

### Tokenization

The same function is applied to the chunks (at indexing time) and to the
questions (at search time), so that their words can match:

1. extract every run of letters, digits and underscores;
2. for each one keep **the whole identifier** and **its parts**, split on
   underscores (`snake_case`) and on case changes (`camelCase`), with acronyms
   and digits kept together;
3. lowercase everything; a word with nothing to split is kept only once.

| Text                     | Words                                          |
|--------------------------|------------------------------------------------|
| `max_model_len`          | `max_model_len`, `max`, `model`, `len`         |
| `AsyncLLMEngine`         | `asyncllmengine`, `async`, `llm`, `engine`     |
| `GPT2Model`              | `gpt2model`, `gpt2`, `model`                   |
| `--tensor-parallel-size` | `tensor`, `parallel`, `size`                   |

Keeping both forms lets a chunk be found both by a question that quotes the
identifier verbatim and by one that paraphrases it. It also makes the
documentation (`--max-lora-rank`) meet the code (`max_lora_rank`).

### Stopwords

Stopwords are removed **from the questions only**. BM25 only scores the words
of the question, so stopwords left in the chunks are never looked up, and the
result is practically the same as removing them on both sides. The list is
hand-written: common English function words are included, while Python
keywords and words that are often identifiers in vLLM (`max`, `model`,
`default`…) are excluded.

### BM25

Chunks are ranked with **BM25**, using the [`bm25s`](https://github.com/xhluca/bm25s)
library, with `k1 = 1.2` and `b = 0.75`. For each word of the question, BM25
combines:

- **IDF** – how rare the word is in the corpus: rare words weigh more;
- **term frequency with saturation (`k1`)** – repetitions count, but less and
  less;
- **length normalization (`b`)** – the same word weighs more in a short chunk
  than in a long one.

### Persistence

The `index` stage saves into `data/processed/`:

- `bm25/` – the BM25 index, saved by `bm25s`;
- `chunks.json` – the list of chunks, wrapped in the pydantic model
  `IndexData`.

The token lists are not saved: they are only needed to build the index.

### Search

The index is loaded once and reused for every question. For a question:
`k <= 0` is rejected, `k` larger than the number of chunks is reduced, and a
question left with no words (empty, only symbols or only stopwords) or with no
word known to the index returns an empty list instead of crashing.

## Performance analysis

*TODO: recall@k will be added once `search_dataset` and the moulinette run
are in place.*

Current measurements (`max_chunk_size = 2000`):

| Metric                            | Value     |
|-----------------------------------|-----------|
| Files indexed                     | 1926      |
| Chunks                            | 35 481    |
| – from Python files               | 34 544    |
| – from Markdown files             | 937       |
| Longest chunk                     | 2000      |
| Chunking + tokenization time      | ~5.6 s    |

**Observation.** Each top-level Python node is a chunk, so every `import`
statement becomes its own tiny chunk. These chunks are favoured by BM25 length
normalization: the query `AsyncLLMEngine` returns five 56-character import
lines instead of the class definition. Grouping consecutive small nodes is the
first planned improvement.

## Design decisions

- **Coordinates and searchable text are separate.** Chunking works on the
  original text so the character ranges stay exact; all normalization
  (lowercase, splitting, stopwords) happens in the tokenizer.
- **Chunkers never open files.** The orchestrator reads each file once and
  passes the text, so the chunkers are easy to test and file reading happens in
  a single function, always in the same way.
- **One function for reading files**, used for indexing and later for
  rebuilding the context of the answers, so character offsets always match.
- **`Chunk` inherits from `MinimalSource`**: a chunk is already a valid
  retrieval result.
- **Files are processed in sorted order**, so the index is reproducible.
- **Lexical retrieval first.** BM25 is fast, needs no model and handles exact
  identifiers well, which matters for questions about code.

## Challenges faced

- **Off-by-one errors on offsets.** `ast` gives 1-based inclusive line
  numbers, while Python slices and the grader use 0-based character offsets
  with an exclusive end. A table of line start offsets solved the conversion.
- **Comments inside code blocks** in the documentation look exactly like
  Markdown headings; tracking fenced blocks was needed.
- **Covering the whole file.** `ast` ignores comments and blank lines, so
  chunks are built from the gaps between nodes rather than from the nodes
  alone.
- **Very small chunk sizes** exhausted memory, which led to the minimum
  chunk size.

## Example usage

*TODO: to be completed with the CLI.*

## Resources

- P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, NeurIPS 2020.
- S. Robertson, H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond*, Foundations and Trends in Information Retrieval, 2009.
- [bm25s documentation](https://github.com/xhluca/bm25s)
- [Python `ast` module](https://docs.python.org/3/library/ast.html)
- [CommonMark specification](https://spec.commonmark.org/) (headings and
  fenced code blocks)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

An AI assistant (Claude) was used as a tutor throughout the project:

- explaining the concepts (RAG, tokenization, TF-IDF and BM25, `ast`, pydantic
  serialization) step by step;
- reviewing the code written by me and pointing out bugs and edge cases, which
  I then fixed myself;
- writing a few small pieces on request: the unit tests in `tests/`, the
  `try`/`except` fallback in the Python chunker, the stopword list and quick
  test scripts.

All the code was read, understood and adapted before being used.