# Use an official PyTorch image as the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \  
    g++ \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxkbcommon-x11-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*
ENV QT_PLUGIN_PATH=/opt/conda/lib/python3.10/site-packages/cv2/qt/plugins/platforms
ENV QT_QPA_PLATFORM=xcb
# Install Python dependencies
RUN pip install --upgrade pip
# Install mmengine (required for mmsegmentation)
RUN pip install mmengine
# Install MMCV with CUDA support
# RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/index.html
# RUN git clone https://github.com/open-mmlab/mmcv.git
# RUN cd mmcv
# WORKDIR ./mmcv
# RUN pip install -r requirements/optional.txt
# RUN pip install -e . -v
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0rc4,<2.1.0" 
# RUN mim install "mmcv>=2.0.0rc4" 
# RUN mim install "mmcv>=[1.1.2, <=[1, 2, 7]"
# Install mmsegmentation
RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git
RUN cd mmsegmentation
WORKDIR ./mmsegmentation
RUN pip install -v -e .
# Verify installation
RUN python -c "import mmseg; print(mmseg.__version__)"
RUN apt update
RUN apt install nano
RUN pip install ftfy regex opencv-python torchinfo tensorboard scikit-image imaug
WORKDIR /app
# Default command (can be overridden when running the container)
CMD ["/bin/bash"]