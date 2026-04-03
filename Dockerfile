# ─── Stage 1: build C++ binary ───────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

# bust stale cache
ARG CACHEBUST=2

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    cmake build-essential git ca-certificates libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt .
COPY src/ src/
COPY include/ include/
COPY bench/ bench/
COPY tests/ tests/

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target heston_demo -j$(nproc) \
    && ls -lh build/heston_demo

# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM ubuntu:22.04

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libfftw3-3 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /app/build/heston_demo build/heston_demo

# Copy Python web layer
COPY web/requirements.txt web/requirements.txt
RUN pip3 install --no-cache-dir -r web/requirements.txt

COPY web/ web/

# Results directory
RUN mkdir -p results

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "web.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
