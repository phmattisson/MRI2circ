# Select the OS
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
# Set user to root
USER root
# Set timezon
ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y tzdata

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --upgrade pip 


RUN apt-get update && apt-get install -y \
        ffmpeg libsm6 libxext6 \
        gcc python3.9-dev

RUN apt-get update && apt-get install -y libhdf5-dev
    
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir  tensorflow==2.10 \
    nilearn pandas \
    scipy numpy matplotlib tqdm imageio scikit-image \
    scikit-learn itk-elastix SimpleITK nibabel intensity-normalization[ants]\
    wandb jupyter opencv-python 
# Verify TensorFlow installation
RUN python3.9 -c "import tensorflow"


COPY scripts /sripts
COPY executables /executables
COPY model /model
COPY shared_data /shared_data


# Default command to run on container start
CMD ["python3.9", "/executables/process_image.py"]