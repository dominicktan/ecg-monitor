# Single-stage Dockerfile for ECG Monitor (Streamlit dashboard)

FROM python:3.12-slim

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (layer caching)
COPY --chown=user pyproject.toml uv.lock ./

# Install dependencies (CPU-only PyTorch configured in pyproject.toml)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code, models, and project files
COPY --chown=user src/ src/
COPY --chown=user models/ models/
COPY --chown=user scripts/ scripts/
COPY --chown=user streamlit_app.py README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

USER user

# Download MIT-BIH data if missing on startup, then run Streamlit
ENTRYPOINT ["scripts/entrypoint.sh"]

ENV PORT=7860
EXPOSE ${PORT}

CMD uv run --no-dev streamlit run streamlit_app.py \
    --server.port=${PORT} --server.address=0.0.0.0 \
    --server.headless=true
