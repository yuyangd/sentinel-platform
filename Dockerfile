FROM rayproject/ray:2.9.0-aarch64

RUN pip3 install --upgrade --no-cache-dir pip poetry wheel
# Separates the virtualenv inside the container
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_CACHE_DIR=/var/cache/poetry
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

USER root
RUN mkdir -p /app

# Set the working directory inside the container
WORKDIR /app

# Copy dependency definition first (for caching layers)
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the application code
# This copies the 'serve' folder into /app/serve
COPY serve/ /app/serve
