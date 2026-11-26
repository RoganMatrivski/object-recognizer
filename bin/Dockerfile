# syntax=docker/dockerfile:1
# Build from the workspace root (the whole workspace is needed):
#   docker build -f bin/Dockerfile -t object-recognizer .
#   docker run -p 3000:3000 -v objrec-data:/data object-recognizer

FROM rust:1-trixie AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# ort's default `download-binaries` fetches a prebuilt (glibc) onnxruntime here.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p object-recognizer-bin \
 && cp target/release/object-recognizer-bin /object-recognizer

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libssl3t64 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 -m app \
    && mkdir /data && chown app /data
COPY --from=build /object-recognizer /usr/local/bin/object-recognizer
USER app
VOLUME /data
EXPOSE 3000
# Optional env: ORT_SEG_THREADS, ORT_EMB_THREADS, RUST_LOG
ENTRYPOINT ["object-recognizer", "--listen", "0.0.0.0:3000", "--data-dir", "/data"]
CMD ["-v"]
