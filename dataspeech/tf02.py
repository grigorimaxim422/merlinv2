from datasets import load_from_disk
# dataset = load_dataset("ylacombe/jenny-tts-tags-6h")
dataset = load_from_disk("_cache_tags_bin")
print("Noise 1st sample:", dataset["train"][0]["noise"])
print("Speaking rate 2nd sample:", dataset["train"][0]["speaking_rate"])
del dataset