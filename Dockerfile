
# Use PyTorch official image with CUDA support as base - this is guaranteed to work
# PyTorch images are well-maintained and include CUDA + cuDNN
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=11.8
ENV CUDNN_VERSION=8

LABEL maintainer="github.com/joephaser/paperspaceenvironment"
LABEL description="PyTorch base with CUDA 11.8 + cuDNN 8, Python and AutoGluon for Paperspace GPU usage"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/joephaser/paperspaceenvironment"
LABEL org.opencontainers.image.description="Complete ML environment for AutoGluon, Hugging Face, and VectorBT on Paperspace Gradient"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        curl \
        git \
        git-lfs \
        build-essential \
        pkg-config \
        software-properties-common \
        python3-dev \
        cmake \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libsndfile1 \
        ffmpeg \
        libcurl4-openssl-dev \
        libomp-dev \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        libfreetype6-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libhdf5-dev \
        libnetcdf-dev \
        unzip \
        vim \
        nano \
        htop \
        tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install TA-Lib C library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && ldconfig

# Ensure pip/tools are modern and install packages in one resolver pass with constraints to avoid incompatibilities
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel \
    # Pin PyTorch stack to match base image (CUDA 11.8)
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    # Write constraints file to prevent accidental upgrades by downstream deps
    && printf "torch==2.1.0\ntorchvision==0.16.0\ntorchaudio==2.1.0\n" > /tmp/constraints.txt \
    && python -m pip install --no-cache-dir -c /tmp/constraints.txt \
        "numpy<2" \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
    "plotly<5" \
        jupyterlab \
        jupyter \
        ipywidgets \
        nbformat \
        numba \
        packaging \
    TA-Lib \
    autogluon \
    autogluon.timeseries \
    "autogluon.tabular[all]" \
        "vectorbt<0.26" \
        transformers \
        accelerate \
        huggingface-hub \
        datasets \
        sentencepiece \
        tokenizers \
        safetensors \
        bitsandbytes \
        statsmodels \
        pmdarima \
        yfinance \
        alpha-vantage \
        quantlib \
    && python -m pip check \
    && python -m pip cache purge

# Copy and set up startup script (must be done as root before user switch)
COPY start-jupyter.sh /usr/local/bin/start-jupyter.sh
RUN chmod +x /usr/local/bin/start-jupyter.sh

# Create a non-root user and workspace (align with Gradient)
ARG USER=paperspace
ARG UID=1000
RUN useradd -m -u ${UID} ${USER} || true
WORKDIR /home/${USER}/workspace
RUN chown -R ${USER}:${USER} /home/${USER}/workspace

USER ${USER}

ENV PATH="/home/${USER}/.local/bin:${PATH}"

# Set Hugging Face cache dirs to the non-root user home so downloaded models are writable
ENV HF_HOME=/home/${USER}/.cache/huggingface
ENV HF_DATASETS_CACHE=/home/${USER}/.cache/huggingface/datasets
ENV TRANSFORMERS_CACHE=/home/${USER}/.cache/huggingface/transformers
ENV HF_HUB_DISABLE_TELEMETRY=1

RUN mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE} \
    && chown -R ${USER}:${USER} /home/${USER}/.cache

# Configure JupyterLab for Paperspace Gradient
RUN mkdir -p /home/${USER}/.jupyter \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> /home/${USER}/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.port = 8888" >> /home/${USER}/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> /home/${USER}/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = True" >> /home/${USER}/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.token = ''" >> /home/${USER}/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.password = ''" >> /home/${USER}/.jupyter/jupyter_lab_config.py \
    && chown -R ${USER}:${USER} /home/${USER}/.jupyter

# Expose JupyterLab port
EXPOSE 8888

# Basic healthcheck for Jupyter (ok if command overridden by platform)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://localhost:8888/ || exit 1

# Default command launches Jupyter (platforms like Gradient can override)
CMD ["start-jupyter.sh"]
