python3 prepare.py "ylacombe/jenny-tts-6h" \
  --configuration "default" \
  --model_name_or_path "google/gemma-2b-it"


# python3 main.py "ylacombe/jenny-tts-6h" \
#   --configuration "default" \
#   --text_column_name "transcription" \
#   --audio_column_name "audio" \
#   --cpu_num_workers 2 \
#   --num_workers_per_gpu_for_pitch 2 \
#   --output_dir "_cache_tags" \
#   --rename_column

##Need to get audio data fc0.pt??