FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

RUN conda init

RUN apt update && apt upgrade -y && apt install screen speedtest-cli git-lfs  ffmpeg -y && rm -rf /var/lib/apt/lists/*

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



