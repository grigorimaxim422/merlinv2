#will create dataset in grigorimaxim/jenny-tts-tags-6h
python3 main.py "ylacombe/jenny-tts-6h" \
  --configuration "default" \
  --text_column_name "transcription" \
  --audio_column_name "audio" \
  --cpu_num_workers 2 \
  --num_workers_per_gpu_for_pitch 2 \
  --output_dir "tmp_cache01" \
  --rename_column
  
#  --rename_column \
#  --repo_id "jenny-tts-tags-6h"

