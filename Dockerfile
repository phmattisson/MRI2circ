FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

#FROM tensorflow/tensorflow:2.10.0-gpu

USER root

ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update && apt install -y tzdata

# Install build dependencies and Python
RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    git \
    cmake \
    zlib1g-dev \
    libpng-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install --no-install-recommends -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-setuptools \
    python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install ITK via pip
RUN pip3 install itk



# Set ITK_DIR so that ITK can be found by CMake
ENV ITK_DIR=/usr/lib/cmake/ITK

# Upgrade pip and install remaining dependencies
RUN python3.9 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir wheel 'numpy<2'


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Add this before pip installations
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir wheel

# Then try installing with binary packages when possible
RUN pip3 install --no-cache-dir --only-binary=:all: \
    tensorflow==2.10 \
    nilearn pandas \
    scipy numpy matplotlib tqdm imageio scikit-image \
    scikit-learn SimpleITK nibabel \
    wandb jupyter

    RUN pip3 install --no-cache-dir shapely


RUN pip3 install --no-cache-dir opencv-python-headless


# Install the ANTs-related packages separately
RUN pip3 install --no-cache-dir itk-elastix
#RUN pip3 install --no-cache-dir intensity-normalization[ants]
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
RUN pip3 install --no-cache-dir --verbose antspyx --no-binary antspyx


COPY scripts /scripts
COPY executables /executables
COPY model /model
COPY shared_data /shared_data
COPY *.py /

# Default command to run on container start
CMD ["python3.9", "/executables/process_integrate.py"]
