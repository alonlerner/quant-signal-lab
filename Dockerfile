FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, from the lockfile only. Docker reuses this layer until
# requirements.lock changes, so editing code doesn't reinstall pandas and friends.
# The lock includes the dev extra so the tests can run inside the image.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Then the project itself. --no-deps because everything it needs is installed above.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY tests ./tests

# Run as a non-root user that owns /app, including the data directory.
RUN useradd --create-home app \
    && mkdir -p data \
    && chown -R app:app /app
USER app

# Sample download. Pass a different command to docker run to override it.
CMD ["signal-lab", "download", "--tickers", "AAPL", "MSFT", "--start", "2024-01-01", "--end", "2024-12-31"]
