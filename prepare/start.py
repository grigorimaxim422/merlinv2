from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from datasets import load_dataset, Audio

DATA_CACHE_DIR="../_cache/"


def download_from_hub(model_repo, cache_dir=DATA_CACHE_DIR):
    print(f"pre-download model {model_repo} started...")

    snapshot_download(repo_id=model_repo, cache_dir=cache_dir)
    print(f"Model {model_repo} downloaded to custom cache directory!")
    
def download_from_hub_to_home(model_repo):
    print(f"pre-download model {model_repo} started...")

    snapshot_download(repo_id=model_repo)
    print(f"Model {model_repo} downloaded to custom cache directory!")

def download_dataset_from_hub(dataset_name,cache_dir=DATA_CACHE_DIR):
    print(f"pre-download dataset {dataset_name} started...")
    dataset = load_dataset(dataset_name,  cache_dir=cache_dir)
    del dataset
    print(f"Dataset {dataset_name} downloaded to custom cache directory!")
    
    
model_repo = "maxrmorrison/fcnf0-plus-plus"
download_from_hub_to_home(model_repo)
model_repo = "ylacombe/brouhaha-best"
download_from_hub(model_repo)

model_repo = "google/gemma-2b-it"
download_from_hub(model_repo)

dataset_name="ylacombe/jenny-tts-6h"
download_dataset_from_hub(dataset_name)




# # Download the whole model (including config and tokenizer)
# model_path = hf_hub_download(repo_id=model_repo)

# # Load model and tokenizer from the downloaded path
# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)


# import os
# from transformers import AutoModel

# DATA_CACHE_DIR="../_cache/"

# model_repo = "maxrmorrison/fcnf0-plus-plus"


# AutoModel.from_pretrained(model_repo, local_files_only=False)

