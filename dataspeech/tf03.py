from datasets import load_dataset
dataset = load_dataset("grigorimaxim/jenny-tts-6h-tagged")
print("1st sample:", dataset["train"][0]["text_description"])
print("2nd sample:", dataset["train"][1]["text_description"])
del dataset