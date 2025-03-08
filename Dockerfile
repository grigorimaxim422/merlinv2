FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

# PM2
RUN apt-get update && apt-get install -y curl \
&& curl -sL https://deb.nodesource.com/setup_18.x | bash - \
&& apt-get install -y nodejs \
&& npm install -g pm2

# Installing necessary packages
RUN apt-get update && apt-get install -y wget git

# Installing Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
&& chmod 700 Miniconda3-latest-Linux-x86_64.sh \
&& ./Miniconda3-latest-Linux-x86_64.sh -b \
&& rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"
# RUN conda create -n venv_image python=3.10.13 -y
# ENV PATH="/root/miniconda3/envs/venv_image/bin:${PATH}"

RUN conda init

RUN apt-get update \
     && apt-get upgrade -y \
     && apt-get -y full-upgrade \
     && apt-get -y install python3-dev \
     && apt-get install -y --no-install-recommends \
     build-essential \
     python3-pip\
     apt-utils \
     curl \
     wget \    
     vim \
     sudo \
     git \
     ffmpeg \
     libsm6 \
     libxext6 \
     python3-tk \
     python3-dev \
     git-lfs \
     unzip \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

#RUN /usr/sbin/update-ccache-symlinks

COPY . /workspace

RUN pip install --upgrade pip \
    && pip install --upgrade pip setuptools \
    && pip install huggingface-hub

RUN python /workspace/login.py


COPY token /root/.huggingface/token
COPY stored_tokens /root/.huggingface/stored_tokens

WORKDIR /workspace/parler-tts

RUN pip install -e .[train] \
    && pip install -r /workspace/dataspeech/requirements.txt \
    && pip install -r /workspace/rvalidator/requirements.txt \
    && pip install --upgrade protobuf \
    && rm -rf /root/.cache/pip/*


RUN git config --global credential.helper store

RUN huggingface-cli whoami

RUN chmod +x /workspace/dataspeech/*.sh && chmod +x /workspace/parler-tts/*.sh && chmod +x /workspace/rvalidator/*.sh && chmod +x /workspace/*.sh

WORKDIR /workspace

RUN cd /workspace && ./prepare.sh

CMD ["sh", "-c", "bin/sh"]



