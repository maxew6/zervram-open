# Stage 1: Pure-Rust build (zero compile-time CUDA/ROCm SDK or OpenSSL headers)
FROM rust:1.85-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
RUN cargo build --release --bin zervramd --bin zervram-scrub

# Stage 2: Minimal runtime mounting host libcuda.so.1 + libnvidia-ml.so.1 via NVIDIA Container Runtime
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/lib/zervram /etc/zervram \
    && chmod 0750 /var/lib/zervram /etc/zervram

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=builder /src/target/release/zervramd /usr/local/bin/zervramd
COPY --from=builder /src/target/release/zervram-scrub /usr/local/bin/zervram-scrub

EXPOSE 9740
ENTRYPOINT ["/usr/local/bin/zervramd"]
CMD ["--listen", "127.0.0.1:9740", "--state-dir", "/var/lib/zervram", "--key-file", "/etc/zervram/node.key"]
