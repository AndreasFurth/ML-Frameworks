FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN pip install --no-cache-dir uv \
    && uv pip install --system --requirements pyproject.toml

COPY . /app

ENV PORT=8000
EXPOSE 8000

# Example: FastAPI server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
