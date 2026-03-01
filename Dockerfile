FROM python:3.12-slim

LABEL maintainer="ollama-forge" \
      description="Batteries-included local AI framework for Ollama"

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir textual fastapi uvicorn jinja2

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Default: run the Web UI on port 8080
EXPOSE 8080

# Can be overridden: forge chat, forge tui, forge api, etc.
ENTRYPOINT ["forge"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8080"]
