# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    ffmpeg

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

RUN mkdir -p /data
RUN mkdir -p /data/models
RUN mkdir -p /data/audio

COPY . .

ENTRYPOINT ["python", "src/api/main.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--model_dir", "/data/models", "--audio_dir", "/data/audio"]