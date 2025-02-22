from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
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

model_repo = "parler-tts/parler_tts_mini_v0.1"
download_from_hub(model_repo)

model_repo = "parler-tts/dac_44khZ_8kbps"
download_from_hub(model_repo)

model_repo = "google/gemma-2b-it"
download_from_hub(model_repo)

model_repo = "google/gemma-2b-it"
download_from_hub(model_repo)

DISCRIMINATOR_FILE_NAME = "discriminator_v1.0.pth"
MODEL_PCA_FILE_NAME = "discriminator_pca_v1.0.pkl"
    
model_path = hf_hub_download(
    repo_id="DippyAI-Speech/Discriminator",
    filename=DISCRIMINATOR_FILE_NAME,  # Replace with the correct filename if different
    cache_dir=DATA_CACHE_DIR
)

# Load the state dictionary into the model
pca_model_path = hf_hub_download(
    repo_id="DippyAI-Speech/PCA", filename=MODEL_PCA_FILE_NAME ,
    cache_dir = DATA_CACHE_DIR# Replace with the correct filename if different
)

dataset_name="DippyAI/dippy_synthetic_dataset"
download_dataset_from_hub(dataset_name)

dataset_name="DippyAI/personahub_augmented_v0"
download_dataset_from_hub(dataset_name)

model_repo = "openai/whisper-tiny"
download_from_hub(model_repo)

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

