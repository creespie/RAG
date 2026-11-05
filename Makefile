# Arguments passed to the program, e.g.:
#   make run ARGS="search 'how to configure lora' --k 5"
ARGS ?=

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 src/*.py
	uv run mypy src/*.py $(MYPY_FLAGS)

lint-strict:
	uv run flake8 src/*.py
	uv run mypy src/*.py --strict