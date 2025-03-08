@@ -0,0 +1,32 @@
 from TTS.api import TTS
 tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
 
 # generate speech by cloning a voice using default settings
 tts.tts_to_file(text="Merci Bous que, sil vous plait.",
                 file_path="output.wav",
                 speaker_wav="/path/to/target/speaker.wav",
                 language="fr")
 
 
 #  tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
 #      --text "Bugün okula gitmek istemiyorum." \
 #      --speaker_wav /path/to/target/speaker.wav \
 #      --language_idx tr \
 #      --use_cuda true
 
 # from TTS.tts.configs.xtts_config import XttsConfig
 # from TTS.tts.models.xtts import Xtts
 
 # config = XttsConfig()
 # config.load_json("/path/to/xtts/config.json")
 # model = Xtts.init_from_config(config)
 # model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
 # model.cuda()
 
 # outputs = model.synthesize(
 #     "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
 #     config,
 #     speaker_wav="/data/TTS-public/_refclips/3.wav",
 #     gpt_cond_len=3,
 #     language="en",
 # )