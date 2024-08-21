# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# ARG: The environment variables only exist in the build time
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION} AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# This user may affect HuggingFace
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# INSTALL FFMPEG: could not drop the apt-get update command
# Put it in the beginning of the file to use caches if possible
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    ffmpeg
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Make directory to mount with local data folder to link to models and data
RUN mkdir -p /files

# Switch to the non-privileged user to run the application.
# Comment this one if the root user access is needed
#USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE ${SPEECH_RECOGNITION_PORT}
# Expose debugger port
EXPOSE 5678

# Run the application.
# Use workdir with relative path before uvicorn sometimes does not work
# Need to use absolute path inside uvicorn to prevent changing of the
# working directory by other systems
# Need to set above src to ensure that src could be used as module
# CMD [..,..], ${SPEECH_RECOGNITION_PORT}

CMD uvicorn 'src.api.main:app' --app-dir=/app \
--host=0.0.0.0 --port=${SPEECH_RECOGNITION_PORT} \
--reload --reload-dir=/app;
RUN echo "TRuntime RECOGNITION PORT is: ${SPEECH_RECOGNITION_PORT}"





# docker run -v /home/data:/data --env model_dir="/home/... --env
# speech_recognition_port=
# use env options in doccker run
