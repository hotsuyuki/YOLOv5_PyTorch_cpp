# Ubuntu 18.04, CUDA 10.2, cuDNN 7.6.5, TensorRT 7.0.0
FROM nvcr.io/nvidia/tensorrt:20.01-py3

ENV DEBIAN_FRONTEND noninteractive

ARG OPENCV_VERSION="4.4.0"
#ARG GPU_ARCH="5.0 5.2 6.1 7.0 7.5 8.6"

WORKDIR /workspace

# Build tools
RUN apt update && \
    apt install -y \
    sudo \
    tzdata \
    git \
    cmake \
    wget \
    unzip \
    build-essential \
    pkg-config

# Media I/O
RUN sudo apt install -y \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libgdal-dev \
    libgtk2.0-dev

# Video I/O
RUN sudo apt install -y \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libopencv-highgui-dev \
    ffmpeg

# Parallelism
RUN sudo apt install -y \
    libtbb-dev 

# # Linear algebra
# RUN sudo apt install -y libeigen3-dev

# Python
RUN sudo apt install -y \
    python3.8 \
    python3.8-venv

# Build OpenCV
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && \
    mkdir build && \
    cd build && \
    cmake \
      -D WITH_TBB=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
      #-D CUDA_ARCH_BIN=${GPU_ARCH} \
      #-D CUDA_ARCH_PTX=${GPU_ARCH} \
      #-D WITH_EIGEN=ON \
      #-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      .. && \
    make all -j$(nproc) && \
    make install