# Single-stage Dockerfile for ECG Monitor (Streamlit dashboard)

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (CPU-only PyTorch configured in pyproject.toml)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code, models, and project files
COPY src/ src/
COPY models/ models/
COPY scripts/ scripts/
COPY streamlit_app.py README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Download MIT-BIH data if missing on startup, then run Streamlit
ENTRYPOINT ["scripts/entrypoint.sh"]

EXPOSE 8501

CMD ["uv", "run", "--no-dev", "streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
