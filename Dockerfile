# Container for the NPHCDA Zero-Dose platform. Runs Streamlit + the full modelling stack.
# Works on a Hugging Face Docker Space (free CPU = ~16 GB RAM, ample for live PyMC/Prophet/spatial)
# and on any Docker host (Render, Railway, Fly, Cloud Run).
FROM python:3.12-slim

# No GPU needed. cxx= keeps PyTensor off the C compiler (we sample with nutpie/numba).
# Cache dirs point at writable /tmp so the container runs as a non-root user on HF Spaces.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTENSOR_FLAGS=cxx= \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# libgomp1: OpenMP runtime used by numba/scipy/scikit-learn. Other geo libs ship self-contained wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Thread-pool budget (placed AFTER pip install so the install layer stays cached).
# OpenBLAS (numpy/scipy), OpenMP (numba/sklearn) and Rayon (nutpie's Rust NUTS) each spawn their
# own pools; unbounded they oversubscribe and segfault (exit 139) during heavy math. Pinning them
# to 1 fixed that on cpu-basic (2 vCPU) but made live sampling crawl. On cpu-upgrade (8 vCPU) we
# give each pool 4 threads: ~4x faster sampling, still bounded so the pools cannot oversubscribe.
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMBA_NUM_THREADS=4 \
    RAYON_NUM_THREADS=4

COPY . .
RUN mkdir -p /tmp/mpl /tmp/numba /tmp/cache && chmod -R 777 /tmp/mpl /tmp/numba /tmp/cache

# Hugging Face Docker Spaces expect the app on port 7860.
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none"]
