FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    nano \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create Python 3.8 environment
RUN conda create -y -n py38 python=3.8
SHELL ["conda", "run", "-n", "py38", "/bin/bash", "-c"]

# Install PyTorch with CUDA 11.7 support in py38
RUN conda run -n py38 pip install torch==2.0.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
RUN conda run -n py38 pip install opencv-python mmengine openmim ftfy regex torchinfo tensorboard py_trees shapely ephem

# Install MMCV and MMSEG
RUN conda run -n py38 mim install "mmcv>=2.0.0rc4,<2.1.0"
RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git
WORKDIR /mmsegmentation
RUN conda run -n py38 pip install -e .

# Verify
RUN conda run -n py38 python -c "import mmseg; print(mmseg.__version__)"

# Workdir
WORKDIR /app

# Default shell into py38 env
#CMD ["conda", "run", "-n", "py38", "/bin/bash"]

