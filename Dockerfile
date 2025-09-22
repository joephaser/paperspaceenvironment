
# Use PyTorch official image with CUDA support as base - this is guaranteed to work
# PyTorch images are well-maintained and include CUDA + cuDNN
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=11.8
ENV CUDNN_VERSION=9

LABEL maintainer="github.com/joephaser/paperspaceenvironment"
LABEL description="PyTorch base with CUDA 11.8 + cuDNN 9, Python and AutoGluon for Paperspace GPU usage"
LABEL version="1.1.0"
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

# Skip TA-Lib C library installation for now - will try to install Python package with pre-compiled wheels

# Ensure pip/tools are modern and install packages in one resolver pass with constraints to avoid incompatibilities
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel \
    # Pin PyTorch stack to match base image (CUDA 11.8) - use existing versions from base image
    && python -c "import torch; print('Using PyTorch version:', torch.__version__)" \
    # Write constraints file to prevent accidental upgrades by downstream deps  
    && python -c "import torch, torchvision, torchaudio; print('torch=='+torch.__version__+'\ntorchvision=='+torchvision.__version__+'\ntorchaudio=='+torchaudio.__version__)" > /tmp/constraints.txt \
    && python -m pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -c /tmp/constraints.txt \
        "numpy<2" \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
    "plotly" \
        jupyterlab \
        jupyter \
        ipywidgets \
        nbformat \
        numba \
        packaging \
    && echo "TA-Lib C library installation skipped - install manually if needed" \
    && python -m pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -c /tmp/constraints.txt \
    autogluon \
    autogluon.timeseries \
    "autogluon.tabular[all]" \
        "vectorbt<0.30" \
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
    && echo "QuantLib installation skipped due to compilation complexity - install manually if needed" \
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
