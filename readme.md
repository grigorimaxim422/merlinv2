## Prepare Envrionment
```
git clone https://github.com/huggingface/dataspeech.git
cd dataspeech
pip install --quiet -r ./dataspeech/requirements.txt

git clone https://github.com/huggingface/parler-tts.git
%cd parler-tts
pip install --quiet -e .[train]

pip install --upgrade protobuf wandb==0.16.6

git config --global credential.helper store
huggingface-cli login
(password)
```
