FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY outputs/ outputs/
COPY data/ data/
# Optional:
# COPY artifacts/ artifacts/

EXPOSE 10000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
