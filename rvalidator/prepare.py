from datasets import load_dataset
import logging
import os
import tempfile
import uuid
import httpx
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import torch
from fastapi import FastAPI, UploadFile, File
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydantic import BaseModel
import io

# Initialize FastAPI app
app = FastAPI()


logger = logging.getLogger(__name__)

DATASET_CACHE_DIR = "../_cache"
hf_token = os.environ.get("HF_TOKEN")

def main():
    DISCRIMINATOR_FILE_NAME = "discriminator_v1.0.pth"
    MODEL_PCA_FILE_NAME = "discriminator_pca_v1.0.pkl"
    
    model_path = hf_hub_download(
        repo_id="DippyAI-Speech/Discriminator",
        filename=DISCRIMINATOR_FILE_NAME,  # Replace with the correct filename if different
        cache_dir=DATASET_CACHE_DIR
    )

    # Load the state dictionary into the model
    pca_model_path = hf_hub_download(
        repo_id="DippyAI-Speech/PCA", filename=MODEL_PCA_FILE_NAME ,
        cache_dir = DATASET_CACHE_DIR# Replace with the correct filename if different
    )
    
    dataset01 = load_dataset("DippyAI/dippy_synthetic_dataset", streaming=True, token=hf_token, cache_dir=DATASET_CACHE_DIR)    
    dataset02 = load_dataset("DippyAI/personahub_augmented_v0", cache_dir=DATASET_CACHE_DIR)
    del dataset01
    del dataset02
    
    #whisper
    # Load the Whisper model and processor from Hugging Face
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", cache_dir=DATASET_CACHE_DIR)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", cache_dir=DATASET_CACHE_DIR)
    del whisper_model
    del whisper_processor
    
    
    #Load Eval score
    
    
    
if __name__ == "__main__":
    main()
 