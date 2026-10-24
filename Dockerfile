FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

USER 1000:1000
EXPOSE 8000

# One process per container. Replica count, not Uvicorn workers, is how this service scales.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
