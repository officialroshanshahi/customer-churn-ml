FROM python:3.14-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy project specification files and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/

# install dependencies
RUN pip install --no-cache-dir .

# expose ports for FastAPI and Gradio
EXPOSE 8000 7860

CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port 8000 & python -m app.gradio"]