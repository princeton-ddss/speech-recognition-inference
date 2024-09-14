# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update --yes && \
apt-get install --yes --no-install-recommends \
ffmpeg

WORKDIR /app
# RUN --mount=type=cache,target=/root/.cache/pip \
#     --mount=type=bind,source=requirements.txt,target=requirements.txt \
#     python -m pip install -r requirements.txt
COPY . .
RUN pip install .

RUN mkdir -p /data
RUN mkdir -p /data/models
RUN mkdir -p /data/audio

ENTRYPOINT ["speech_recognition_launcher", "--audio_dir", "/data/audio", "--model_dir", "/data/models"]
CMD []
