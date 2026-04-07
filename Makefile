.PHONY: run stop app api data test lint clean

# Docker
run:  ## Start Streamlit dashboard (Docker)
	docker compose up --build

stop:  ## Stop Docker services
	docker compose down

# Local development (no Docker)
app:  ## Run Streamlit dashboard locally
	uv run streamlit run streamlit_app.py

api:  ## Run FastAPI backend locally (standalone, not needed for dashboard)
	uv run uvicorn ecg_monitor.api:app --port 8000 --reload

# Data
data:  ## Download MIT-BIH database from PhysioNet
	uv run python scripts/download_data.py

# Quality
test:  ## Run tests
	uv run pytest tests/ -v

lint:  ## Run linter
	uv run ruff check src/ streamlit_app.py

format:  ## Auto-format code
	uv run ruff format src/ streamlit_app.py

# Cleanup
clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
