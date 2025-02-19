from datasets import load_dataset
dataset = load_dataset("ylacombe/jenny-tts-tags-6h")
print("Noise 1st sample:", dataset["train"][0]["noise"])
print("Speaking rate 2nd sample:", dataset["train"][0]["speaking_rate"])
del dataset