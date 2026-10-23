# The upstream chromadb/chroma image is a Debian-slim Rust image and
# ships neither python nor curl. Compose needs an HTTP probe for the
# heartbeat endpoint, so add curl to a thin layer.
FROM chromadb/chroma:1.5.9

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
