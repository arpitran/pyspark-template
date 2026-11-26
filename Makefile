# Spark needs Java 17+. Homebrew's openjdk@17 is not registered with java_home, so point at it directly.
# Override with: make test JAVA_HOME=/path/to/jdk
JAVA_HOME ?= $(shell /usr/libexec/java_home -v 17 2>/dev/null || echo /opt/homebrew/opt/openjdk@17)
export JAVA_HOME

RUN = uv run python

.PHONY: install data test lint fmt run-write run-debug run-optimize run-all clean

install:
	uv sync

data:
	$(RUN) data/generate.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

run-write:
	@for script in examples/write/*.py; do echo "\n>>> $$script"; $(RUN) $$script || exit 1; done

run-debug:
	@for script in examples/debug/*.py; do echo "\n>>> $$script"; $(RUN) $$script || exit 1; done

run-optimize:
	rm -f results/benchmarks.md
	@for script in examples/optimize/*.py; do echo "\n>>> $$script"; $(RUN) $$script || exit 1; done

run-all: run-write run-debug run-optimize

clean:
	rm -rf output spark-warehouse spark-events metastore_db derby.log .pytest_cache
