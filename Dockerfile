
# Use NVIDIA CUDA runtime base image so container has CUDA & cuDNN available for Paperspace
# Picking CUDA 12.4 and cuDNN 8 to match recent Autogluon GPU requirements (host must provide NVIDIA drivers)
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.4
ENV CUDNN_VERSION=8

LABEL maintainer="github.com/yourusername/paperspace-autogluon-env"
LABEL description="Ubuntu 24.04 LTS with CUDA 12.4 + cuDNN 8 runtime, Python and AutoGluon for Paperspace GPU usage"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/yourusername/paperspace-autogluon-env"
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
        python3 \
        python3-venv \
        python3-pip \
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
    && rm -rf /var/lib/apt/lists/*

# Ensure pip/tools are modern and install the latest stable Autogluon and other Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install core data science and ML packages
RUN python3 -m pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        plotly \
        jupyterlab \
        jupyter \
        ipywidgets \
        nbformat

# Install AutoGluon with GPU support for time series
RUN python3 -m pip install --no-cache-dir \
        autogluon \
        autogluon.timeseries \
        autogluon.tabular[all]

# Install VectorBT and its dependencies
RUN python3 -m pip install --no-cache-dir \
        vectorbt \
        numba \
        talib-binary

# Install Hugging Face / transformers stack for running models locally
RUN python3 -m pip install --no-cache-dir \
        transformers \
        accelerate \
        huggingface-hub \
        datasets \
        sentencepiece \
        tokenizers \
        safetensors \
        bitsandbytes \
        torch \
        torchvision \
        torchaudio

# Install additional time series and financial analysis tools
RUN python3 -m pip install --no-cache-dir \
        statsmodels \
        pmdarima \
        yfinance \
        alpha-vantage \
        quantlib

# Create a non-root user and workspace
ARG USER=gradient
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

# Copy and set up startup script
COPY start-jupyter.sh /usr/local/bin/start-jupyter.sh
RUN chmod +x /usr/local/bin/start-jupyter.sh

# Expose JupyterLab port
EXPOSE 8888

# Default workdir and command
CMD ["bash"]
