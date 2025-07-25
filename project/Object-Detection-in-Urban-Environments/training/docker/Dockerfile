FROM tensorflow/tensorflow:2.13.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    libgl1-mesa-dev \
    wget

RUN pip install 'protobuf<=3.20.1' --force-reinstall
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy this version of the model garden into the image
COPY models/research/object_detection /home/tensorflow/models/research/object_detection 

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
RUN sed -i 's/pillow/pillow==9.5/g' setup.py
ENV PATH="/home/tensorflow/.local/bin:${PATH}"
RUN python -m pip install -U pip
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

# Install SageMaker training-toolkit
RUN pip3 install sagemaker-training
RUN pip install "opencv-python-headless<4.3"

