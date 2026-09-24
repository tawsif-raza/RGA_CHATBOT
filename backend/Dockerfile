# Built from the project root: docker build -f backend/Dockerfile .
FROM python:3.11-slim

# Microsoft ODBC Driver 18 for SQL Server: pyodbc installs fine via pip
# without this, but backend.database's connections would fail at runtime
# with "Data source name not found" - the driver itself is a separate
# native dependency, same as the winget install used for local dev on Windows.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
    && curl -sSL -O https://packages.microsoft.com/config/debian/$(. /etc/os-release && echo "$VERSION_ID")/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc-dev \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Hugging Face model cache (BGE embeddings, BGE reranker, base Qwen weights)
# lives outside the image so a redeploy doesn't re-download multi-GB model
# weights every startup - mount a host folder at this path (see
# docker-compose.prod.yml's `./models_cache:/app/model_cache`).
ENV HF_HOME=/app/model_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
