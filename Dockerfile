FROM python:3.10-slim

WORKDIR /app

# Install runtime deps first (layer cache)
COPY requirements.txt pyproject.toml README.md ./
COPY env/ env/
COPY server/ server/
COPY openenv.yaml inference.py ./
COPY frontend/ frontend/

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]