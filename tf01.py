from datasets import load_dataset
dataset = load_dataset("ylacombe/jenny-tts-6h")

from IPython.display import Audio
print(dataset["train"][0]["transcription"])
Audio(dataset["train"][0]["audio"]["array"], rate=dataset["train"][0]["audio"]["sampling_rate"])

from datasets import load_dataset
dataset = load_dataset("ylacombe/jenny-tts-tags-6h")
print("SNR 1st sample", dataset["train"][0]["snr"])
print("C50 2nd sample", dataset["train"][0]["c50"])
del dataset