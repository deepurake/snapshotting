FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch transformers accelerate "jinja2>=3.1.0"

WORKDIR /app
COPY infer.py .

ENTRYPOINT ["python3", "infer.py"]
CMD ["--auto-loop"]
