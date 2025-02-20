import torch
from fastapi import FastAPI, UploadFile, File
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydantic import BaseModel
import io

DATASET_CACHE_DIR = "evalsets"

# Initialize FastAPI app
app = FastAPI()

# Load the Whisper model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en", cache_dir=DATASET_CACHE_DIR)
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", cache_dir=DATASET_CACHE_DIR)

# API to transcribe audio
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read the audio file from the request
    audio_data = await file.read()
    
    # Convert the audio data into a format that Whisper can process
    audio_input = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values

    # Get the model predictions
    with torch.no_grad():
        predicted_ids = model.generate(audio_input)

    # Decode the generated tokens into text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return {"transcription": transcription[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8007)
