.PHONY: install lint test run mlflow clean

install:
	poetry install

lint:
	poetry run ruff check src/
	poetry run ruff format --check src/

lint-fix:
	poetry run ruff check --fix src/
	poetry run ruff format src/

test:
	poetry run pytest src/tests/ -v --tb=short

run:
	poetry run python -c "from src.config import setup_logging; setup_logging(); from src.pipeline import ChurnPipeline; ChurnPipeline().run()"

MLFLOW_EXE ?= $(shell poetry env info --path 2>/dev/null)/Scripts/mlflow.exe

mlflow:
	$(MLFLOW_EXE) ui --backend-store-uri sqlite:///mlruns.db --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
