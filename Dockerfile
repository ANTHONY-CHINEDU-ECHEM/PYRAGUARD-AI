FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api,dashboard]"

COPY configs ./configs
COPY knowledge_base ./knowledge_base
COPY data/eval ./data/eval

# build the knowledge index into the image so the first request is fast
RUN pyraguard index

RUN useradd --create-home pyraguard && chown -R pyraguard /app
USER pyraguard

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["pyraguard", "serve", "0.0.0.0", "8000"]
