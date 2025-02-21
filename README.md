## Prepare Envrionment

[Colab] https://colab.research.google.com/github/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb#scrollTo=LKwAePwUcl8u

[Origin Medium article] https://blog.gopenai.com/getting-started-with-parler-tts-tips-for-fine-tuning-and-inference-1911171b2e5a

<!-- ```
git clone https://github.com/huggingface/dataspeech.git
cd dataspeech
pip install --quiet -r ./dataspeech/requirements.txt

git clone https://github.com/huggingface/parler-tts.git
%cd parler-tts
pip install --quiet -e .[train] -->

apt install ffmpeg -y
cd parler-tts
pip install  -e .[train]
pip install  -r ../dataspeech/requirements.txt
pip install  -r ../rvalidator/requirements.txt
pip install --upgrade protobuf wandb==0.16.6

git config --global credential.helper store
huggingface-cli login
(password)
```

cd dataspeech
chmod +x *.sh

./01.create_dataset.sh

python3 tf01.py

./02.map_anno_2_textbin.py

python3 tf02.py

./03.create_nld_from_textbin.sh

cd ..
cd parler-tts

chmod +x *.sh
./04.finetune.sh


## How to upload to huggingface 
- Preparation
```
pip install huggingface_hub
huggingface-cli login
```
- Upload
```

huggingface-cli repo create your-model-name
git lfs clone https://huggingface.co/your-username/your-model-name
cd your-model-name
git lfs install
git lfs track "*.safetensors"
git lfs track "tokenizer.json"
git add .gitattributes
git commit -m "init"
git push

git add .
git commit -m "Add trained model files"
git config user.name grigorimaxim
git config user.email grigorimaxim@gmail.com
git push origin main
```