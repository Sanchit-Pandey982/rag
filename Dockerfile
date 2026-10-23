FROM python:3.11-slim

# Optional PyPI mirror for slow networks. The default is the official index;
# override at build time with:
#   docker compose build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ backend
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /app

COPY requirements.txt .

# Do not upgrade pip here: it is one more large download from the same
# index, and the base image's pip is sufficient. Keep the cache mount so
# retries reuse already-downloaded wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && attempt=1 \
    && while ! pip install --index-url "$PIP_INDEX_URL" -r requirements.txt; do \
         attempt=$((attempt + 1)); \
         if [ "$attempt" -gt 10 ]; then \
           echo "pip install failed after 10 attempts" >&2; \
           exit 1; \
         fi; \
         echo "pip install failed; retrying ($attempt/10)..." >&2; \
         sleep 5; \
       done

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
