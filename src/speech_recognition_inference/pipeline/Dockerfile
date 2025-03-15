# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update --yes && \
apt-get install --yes --no-install-recommends \
ffmpeg

WORKDIR /app
COPY . .
RUN pip install .

RUN mkdir -p /data
RUN mkdir -p /data/models
RUN mkdir -p /data/audio

ENTRYPOINT ["speech_recognition_pipeline", "/data/audio", "--model-dir", "/data/models"]
