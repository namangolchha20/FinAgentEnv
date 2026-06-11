FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY env/ env/
COPY server/ server/
COPY openenv.yaml ./
COPY frontend/ frontend/

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
