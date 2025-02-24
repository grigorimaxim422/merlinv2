# Prelog

This project aims to fine-tune the current parler-TTS model to improve it. We evaluate the fine-tuned model in two dimensions.
- We evaluate the WER(Word Error Rate) score using whisper speech recognition Inference.
- We evaluate the Human Similarity Score using custom code

You can compare the evaluation result with the original parler-TTS model, how much you improved it.

## Useful resources
- [Docker] https://hub.docker.com/r/grigorimaxim/merlin_tts  (53.8GB)
- [Github] https://github.com/grigorimaxim422/merlinv2
- [DockerImage] https://huggingface.co/grigorimaxim/merlin_tts/resolve/master/merlin_tts.tar.gz?download=true  (40GB)
- [Colab] https://colab.research.google.com/github/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb#scrollTo=LKwAePwUcl8u
- [Medium] https://blog.gopenai.com/getting-started-with-parler-tts-tips-for-fine-tuning-and-inference-1911171b2e5a

# How to use?

## Pre1: How to use docker image

### From online
```
sudo docker pull grigorimaxim/merlin_tts
sudo docker save -o merlin_tts.tar grigorimaxim/merlin_tts:latest

```

## load and run from pre-downloaded file
```
#unzip image file
gunzip -c merlin_tts.tar.gz > merlin_tts.tar
#Load docker image
sudo docker load < merlin_tts.tar
#run docker image
sudo docker run --gpus all --rm -it grigorimaxim/merlin_tts
```
## How to run this project in docker image
- Prepare the dataset to train
```
cd dataspeech 
./01.create_dataset.sh
./02.map_anno_2_textbin.sh
./03.create_nld_from_textbin.sh
cd ..
```
- Train the dataset to fine tune the parler-tts model
```
cd parler-tts
./04.finetune.sh
cd ..
```
- To validate the fine-tuned model
```
cd rvalidator
./05.validate.sh
```

# Contribute

## How to install this project?
```
sudo apt update && sudo apt upgrade -y
git clone https://github.com/grigorimaxim422/merlinv2
git checkout dev

# Install dependencies
sudo apt install ffmpeg -y
pip install  -e .[train]
pip install  -r ../dataspeech/requirements.txt
pip install  -r ../rvalidator/requirements.txt
pip install --upgrade protobuf wandb==0.16.6
git config --global credential.helper store

# huggingface login
huggingface-cli login
```

