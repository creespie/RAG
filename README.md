*This project has been created as part of the 42 curriculum by lurossi.*

# RAG against the machine

## Description

This project is a **Retrieval-Augmented Generation (RAG)** system that answers
questions about a codebase: the source code and documentation of
[vLLM](https://github.com/vllm-project/vllm) 0.10.1.

A language model only knows what it saw during training. Instead of retraining
it, a RAG system looks up the relevant information at question time and hands
it to the model together with the question. The pipeline has four stages:

1. **Indexing** – split every file of the corpus into chunks and build a
   searchable BM25 index.
2. **Retrieving** – given a question, find the most relevant chunks.
3. **Augmenting** – place those chunks in the model's context, within a token
   budget.
4. **Generating** – let a small local model (`Qwen/Qwen3-0.6B`, on CPU) write
   an answer grounded in them.

Retrieval quality is measured with recall@k: for each question, the system
must return the right **source locations** (file path and character range).
On the public datasets it reaches **82.0% recall@5 on documentation questions
and 73.7% on code questions** (targets: 80% and 50%).

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) as package and project manager
- About 3 GB of free disk space for the dependencies (CPU-only PyTorch) and
  the model weights, downloaded on first use into the Hugging Face cache

`pyproject.toml` pins PyTorch to the CPU-only index, so `uv sync` does not
download the CUDA libraries.

### Disk space on campus machines

The dependencies (CPU-only PyTorch, Transformers) and the model weights take
about 3 GB, which does not fit in a small home directory. Before installing,
point the caches to a larger local disk such as `/goinfre` (the exact path
depends on the campus):

```bash
export UV_CACHE_DIR=/goinfre/$USER/uv_cache   # packages downloaded by uv
export HF_HOME=/goinfre/$USER/hf_cache        # Qwen3-0.6B weights (~1.5 GB)
export UV_LINK_MODE=copy                      # cache and project on different disks
echo $UV_CACHE_DIR $HF_HOME $UV_LINK_MODE     # check
```

Add these lines to `~/.zshrc` (or `~/.bashrc`) to keep them in every new
terminal, then open a new terminal or run `source ~/.zshrc`.

If the home directory is still too small, the virtual environment can be
moved as well:

```bash
export UV_PROJECT_ENVIRONMENT=/goinfre/$USER/rag_venv
```

or the whole repository can be cloned directly under `/goinfre`.

`/goinfre` may be wiped between sessions: in that case `make install`
reinstalls the dependencies and the model is downloaded again on the first
`answer` command.

### Installation

```bash
make install        # runs: uv sync
```

The vLLM repository must be placed in `data/raw/` (e.g.
`data/raw/vllm-0.10.1/`) and the question datasets in `data/datasets/`.
The `data/` folder is not part of the repository.

### Makefile rules

| Rule          | Description                                                 |
|---------------|-------------------------------------------------------------|
| `install`     | Install the dependencies with `uv sync`                     |
| `run`         | Run the program; arguments go in `ARGS`                     |
| `debug`       | Run the program under `pdb`; arguments go in `ARGS`         |
| `clean`       | Remove caches (`__pycache__`, `.mypy_cache`, `.pytest_cache`) |
| `lint`        | `flake8` and `mypy` with the required flags                 |
| `lint-strict` | `flake8` and `mypy --strict`                                |

Example: `make run ARGS="search 'how to configure lora' --k 5"`.

### Commands

Every command is run as `uv run python -m src <command> [options]`.

| Command          | Options (defaults)                                                        | Output |
|------------------|---------------------------------------------------------------------------|--------|
| `index`          | `--max_chunk_size 2000` `--input_path data/raw` `--output_path data/processed` | index saved in `data/processed/` |
| `search`         | `<query>` `--k 10` `--index_folder data/processed`                        | top-k sources printed |
| `search_dataset` | `--dataset_path` `--save_directory` `--k 10` `--index_folder`              | `StudentSearchResults` JSON |
| `answer`         | `<query>` `--k 5` `--index_folder` `--max_context_tokens 2000`            | answer and sources printed |
| `answer_dataset` | `--student_search_results_path` `--save_directory` `--max_context_tokens 2000` | `StudentSearchResultsAndAnswer` JSON |
| `evaluate`       | `--student_search_results_path` `--dataset_path`                         | recall@1/3/5/10 |

The output file of `search_dataset` and `answer_dataset` has the same name as
the input file. `index` must be run before the other commands; otherwise they
print `Index not found ..., run the index command first`.

## System architecture

```
                    ┌──────────── index ────────────┐
data/raw/ ─► corpus ─► chunking ─► tokenizer ─► BM25 (bm25s) ─► data/processed/
                          ▲            ▲                         (bm25/ + chunks.json)
                       headings ───────┘                               │
                                                                       ▼
question ─► tokenizer + stopwords ─► retriever ─► top-k sources ─► generator ─► answer
                                                      │              (Qwen3-0.6B)
                                                      └─► evaluation (recall@k)
```

| Module          | Responsibility |
|-----------------|----------------|
| `__main__.py`   | CLI (Python Fire): input checks, error messages, orchestration of the commands. |
| `corpus.py`     | Lists the `.py`, `.md` and `.txt` files of the corpus (skipping hidden folders), in sorted order, and reads them as UTF-8, skipping unreadable files. |
| `chunking.py`   | Splits a file into chunks: Python strategy, Markdown strategy and line-based fallback. |
| `headings.py`   | Finds the Markdown section titles that contain each chunk. |
| `tokenizer.py`  | Turns a text into a list of words; holds the stopword list. |
| `indexer.py`    | Builds the chunks and the BM25 index, saves and loads them. |
| `retriever.py`  | Returns the top-k chunks for a question. |
| `evaluation.py` | IoU, recall@k and a diagnosis of the failed questions. |
| `generator.py`  | Builds the context and the prompt, and generates the answer with Qwen. |
| `models.py`     | Pydantic models: those of the subject, plus `Chunk` and `IndexData`. |

Each module does one thing and receives its input as parameters: the chunkers
work on a string and never open files, so they can be tested on hand-written
inputs. The index and the model are loaded **once** per command and reused for
every question.

## Chunking strategy

A chunk is a file path and a character range (`first_character_index`,
`last_character_index`, end excluded). The chunks of a file are
**contiguous and cover the whole file**: every character belongs to exactly one
chunk, so no part of the corpus is invisible to retrieval. No chunk is ever
longer than `max_chunk_size`, which must be at least 100.

### Line-based fallback

Lines, each keeping its trailing `\n` so that lengths add up exactly, are
accumulated until the next one would exceed the limit. A single line longer
than the limit is cut into pieces of `max_chunk_size` characters. It is used
for oversized pieces, for Python files that cannot be parsed, and for `.txt`
files (e.g. `CMakeLists.txt`), which have no structure to follow.

### Python files

The file is parsed with the standard `ast` module. Each chunk goes from the
end of the previous top-level node to the end of the current one, so
decorators, blank lines and comments placed before a function or class stay
with it, and trailing comments form a final chunk. Oversized nodes are split
with the line-based fallback. If the file cannot be parsed (`SyntaxError`, or
`ValueError` for null bytes), the whole file is split by lines.

### Markdown files

The file is split **before** each heading of level 1 or 2 (`#`, `##`). A line
is a heading if it has at most 3 leading spaces, 1 or 2 `#` characters, then a
space or a tab. Two details:

- **Fenced code blocks** (```` ``` ```` and `~~~`, also when indented inside
  MkDocs tabs) are tracked, so a shell or Python comment such as
  `# install vllm` is not mistaken for a heading. A block is closed only by
  the same kind of fence that opened it.
- **Consecutive headings with no content between them** stay together, so a
  page title is not left alone in a one-line chunk.

Oversized sections are split with the line-based fallback.

## Retrieval method

### Tokenization

The same function is applied to the chunks (at indexing time) and to the
questions (at search time), so that their words can match:

1. extract every run of letters, digits and underscores;
2. for each one keep **the whole identifier** and **its parts**, split on
   underscores (`snake_case`) and on case changes (`camelCase`), keeping
   acronyms and digits together;
3. lowercase everything; a word with nothing to split is kept only once.

| Text                     | Words                                      |
|--------------------------|--------------------------------------------|
| `max_model_len`          | `max_model_len`, `max`, `model`, `len`     |
| `AsyncLLMEngine`         | `asyncllmengine`, `async`, `llm`, `engine` |
| `GPT2Model`              | `gpt2model`, `gpt2`, `model`               |
| `--tensor-parallel-size` | `tensor`, `parallel`, `size`               |

Keeping both forms lets a chunk be found both by a question that quotes an
identifier verbatim and by one that paraphrases it, and makes the
documentation (`--max-lora-rank`) meet the code (`max_lora_rank`).

### Enriching the indexed text

The words indexed for a chunk are **not only its own**. The character range
returned to the grader is unchanged, but BM25 also sees:

- **the words of the file path**: `docs/features/quantization/quark.md` adds
  `features`, `quantization`, `quark`;
- **for Markdown, the titles of the sections containing the chunk**: a piece in
  the middle of a long section still carries the page title and the section
  title.

Words common to every path (`data`, `raw`, `vllm`) get an almost zero IDF and
do no harm.

### Stopwords

Stopwords are removed **from the questions only**. BM25 only scores the words
of the question, so stopwords left in the chunks are never looked up. The list
is hand-written: common English function words are included, while words that
are often identifiers in vLLM (`max`, `model`, `default`…) are not.

### BM25

Chunks are ranked with **BM25** through the
[`bm25s`](https://github.com/xhluca/bm25s) library, with `k1 = 1.2` and
`b = 0.75`. For each word of the question, BM25 combines:

- **IDF** – how rare the word is in the corpus: rare words weigh more;
- **term frequency with saturation (`k1`)** – repetitions count, but less and
  less;
- **length normalization (`b`)** – the same word weighs more in a short chunk
  than in a long one.

### Persistence and search

`index` saves `data/processed/bm25/` (the index, saved by `bm25s`) and
`data/processed/chunks.json` (the chunks, wrapped in the pydantic model
`IndexData`). The token lists are not saved: they are only needed to build the
index.

At search time, `k <= 0` is rejected, a `k` larger than the number of chunks
is reduced, and a question left with no words (empty, only symbols or only
stopwords) or with no word known to the index returns an empty list instead
of crashing.

## Answer generation

### Context

The retrieved sources are read back from the files (each file is read once and
cached), formatted as `[Source N] <file path>` followed by their text, and
added **best first** until a budget of 2000 tokens, counted with the model's
own tokenizer. If no source is found, the model is not called at all and a
fixed "could not find" answer is returned.

### Prompt strategy

- A **system message** with the rules: use only the context, never guess
  names or values, answer in at most 3 sentences, quote at most one short line
  of code, say "I could not find the answer in the provided sources." only
  when the context contains nothing relevant.
- A **user message** with the context first and the question **repeated at
  the end**, followed by "Answer using only the context above": a small model
  answers better when the question is the last thing it reads.
- Qwen3 *thinking mode* is disabled (`enable_thinking=False`), which is
  faster on CPU and gives direct answers; any leftover `<think>` block is
  removed.
- Sampling with `temperature=0.3`, `top_p=0.8`, `top_k=20`, at most 256 new
  tokens. Qwen discourages pure greedy decoding, which can loop; a low
  temperature keeps the model close to the wording of the context.

### Effect of the prompt changes

| Question | First prompt (`temperature=0.7`) | Final prompt (`temperature=0.3`) |
|----------|-----------------------------------|----------------------------------|
| What hardware platforms does vLLM support? | Lists the platforms, then adds "The answer is not found in the provided context." | "vLLM supports NVIDIA GPUs, AMD GPUs, Intel GPUs, TPU, and CPU." |
| How do you use GPTQModel quantized models with vLLM's Python API? | Copies a whole code example, cut in the middle by the token limit | A two-sentence explanation |

**Known limitation.** When retrieval misses the right section, the 0.6B model
still tends to produce a plausible but wrong answer instead of admitting it
does not know (e.g. it names a wrong interface for multimodal models, whose
documentation section was not retrieved). Better retrieval is the most
effective fix, since these errors only appear on questions whose sources were
not found.

## Performance analysis

### Recall@5 on the public datasets

Each line adds one change to the previous one (`max_chunk_size = 2000`, `k = 10`).

| Experiment                         | Docs R@5  | Code R@5  |
|------------------------------------|-----------|-----------|
| Baseline (`.py` and `.md` only)    | 74.0%     | 64.6%     |
| + `.txt` files (`CMakeLists.txt`)  | 77.0%     | 64.6%     |
| + words of the file path           | 79.0%     | **73.7%** |
| + Markdown section titles          | **82.0%** | **73.7%** |
| *Target*                           | *80%*     | *50%*     |

Full results of the final version:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 0.620    | 0.760    | 0.820    | 0.880     |
| Code    | 0.444    | 0.636    | 0.737    | 0.859     |

Observations:

- `.txt` files were added after the diagnosis showed that some reference
  sources are in `CMakeLists.txt`, which could not be found at all.
- The file path gave the largest gain on code questions: vLLM file and folder
  names describe their content (`vllm/lora/peft_helper.py`) and questions often
  use those words.
- Section titles only affect Markdown chunks, so they only improved
  documentation questions.
- The average reference source is 1135 characters long and the average
  retrieved chunk 1044, so chunk size was not the bottleneck: most misses were
  "right file, wrong section".

### Speed

| Step                                      | Time              |
|-------------------------------------------|-------------------|
| Chunking + tokenization of 1952 files     | ~6 s (35 565 chunks) |
| `search_dataset`, 100 questions           | < 1 s             |
| Loading Qwen3-0.6B (after the download)   | ~20 s, once per command |
| `answer_dataset`, 100 questions (CPU)     | *TODO*            |

Both limits are met with a wide margin: indexing must take at most 5 minutes,
and 200 questions must be searched in at most 90 seconds.

### Known weaknesses

- Every top-level Python statement is its own chunk, so each `import` line is
  a tiny chunk. BM25 length normalization favours them: the query
  `AsyncLLMEngine` returns 56-character import lines instead of the class
  definition. Grouping consecutive small nodes is the next planned improvement.
- The documentation recall is only 2 points above the target, so a different
  set of questions could fall below it.

## Design decisions

- **Coordinates and searchable text are separate.** Chunking works on the
  original text so that character ranges stay exact; normalization and
  enrichment only change the words given to BM25.
- **Chunkers never open files.** The indexer reads each file once and passes
  the text; file reading happens in a single function, always in the same
  way, so offsets match between indexing and answer generation.
- **`Chunk` inherits from `MinimalSource`**: a chunk is already a valid
  retrieval result.
- **Lexical retrieval with BM25.** Fast, no model needed, and good at exact
  identifiers, which matter for questions about code.
- **Data-driven choices.** File types, enrichment and prompt changes were
  chosen by looking at failed questions with the diagnosis tool
  (`python -m src.evaluation <results> <answered dataset>`) and kept only when
  recall improved.
- **Errors never show a traceback.** Expected errors (missing files, malformed
  JSON, invalid `k`) print `Error: ...`; anything else prints
  `Unexpected error: ...`, so a bug is still distinguishable from a user
  mistake.

## Challenges faced

- **Off-by-one errors on offsets.** `ast` gives 1-based, inclusive line
  numbers, while Python slices and the grader use 0-based character offsets
  with an exclusive end. A table of line start offsets solved the conversion.
- **Covering the whole file.** `ast` ignores comments and blank lines, so
  Python chunks are built from the gaps between nodes rather than from the
  nodes alone.
- **Comments inside code blocks** of the documentation look exactly like
  Markdown headings; fenced blocks had to be tracked.
- **Very small chunk sizes** (e.g. 1) exhausted memory, which led to the
  minimum chunk size.
- **Disk space.** The default PyTorch wheel includes CUDA libraries and does
  not fit in a small home directory; the CPU-only index and moving the caches
  (`UV_CACHE_DIR`, `HF_HOME`) to a larger disk solved it.
- **Hallucinations of a small model.** The first prompt produced
  contradictory answers and long copied code; stricter rules and a lower
  temperature fixed most of it.

## Example usage

```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 2000

# 2. Search a single question
uv run python -m src search "how to configure lora rank" --k 5
# data/raw/vllm-0.10.1/docs/features/lora.md [14472:15148]
# ...

# 3. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions

# 4. Measure recall (own implementation, same numbers as the moulinette)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Answer a single question
uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter?" --k 5
# Answer:
# The HTTP endpoint used to dynamically load a LoRA adapter is `/v1/load_lora_adapter`.

# 6. Answer a whole dataset
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
## Evaluation workflow

### Measuring recall

After `index`, search both public datasets:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

Then score them with the moulinette (official evaluation; results file
first, ground-truth dataset second):

```bash
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json \
    --k 10 --max_context_length 2000
```

The `evaluate` command computes the same recall with this project's own code
and never calls the moulinette.

### Diagnosing failed questions

```bash
uv run python -m src.evaluation <results.json> <answered_dataset.json> [k] [show]
```

Prints recall@1/3/5/10, the length of the reference and retrieved sources,
the share of `.py` and `.md` results, the missed sources grouped by reason
(file never retrieved, right file but wrong section, right chunk ranked after
position k…) and the first `show` failed questions at cut-off `k`
(defaults: 5 and 15), with their top results and IoU.

### Running an experiment

1. change one thing (chunking, tokenization, enrichment, `k1`, `b`…);
2. rebuild the index with `index`;
3. run both `search_dataset` commands and both moulinette commands;
4. keep the change only if recall@5 improves, and record it in the
   performance table.

### Tests

```bash
uv run python -m pytest tests/
```

Unit tests of the chunkers: every chunk respects the size limit, the chunks
of a file are contiguous and cover it entirely, and edge cases (syntax
errors, null bytes, empty files, very long lines) do not crash.

## Resources

- P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, NeurIPS 2020.
- S. Robertson, H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond*, Foundations and Trends in Information Retrieval, 2009.
- N. F. Liu et al., *Lost in the Middle: How Language Models Use Long
  Contexts*, TACL 2024.
- [bm25s](https://github.com/xhluca/bm25s)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) (chat
  template, thinking mode, recommended sampling parameters)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Python `ast` module](https://docs.python.org/3/library/ast.html)
- [CommonMark specification](https://spec.commonmark.org/) (headings and
  fenced code blocks)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire](https://github.com/google/python-fire)
- [uv documentation](https://docs.astral.sh/uv/), including the guide on
  using PyTorch with uv

### Use of AI

An AI assistant (Claude) was used as a tutor throughout the project:

- **explanations**: RAG, tokenization, TF-IDF and BM25, `ast`, pydantic
  serialization, Fire, Hugging Face generation, explained step by step before
  writing the code;
- **code review**: the corpus reader, the chunkers, the tokenizer, the indexer
  and the retriever were written by me and reviewed by the assistant, which
  pointed out bugs and edge cases that I then fixed;
- **code written by the assistant on request**, then read and adapted by me:
  the unit tests in `tests/`, the `try`/`except` fallback of the Python
  chunker, the stopword list, the Makefile and a first draft of this
  README;
- **diagnosis**: interpreting the recall results and the failed questions to
  choose the next experiment.

All the code was read and understood before being used.