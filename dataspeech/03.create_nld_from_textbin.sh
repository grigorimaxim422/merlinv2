python ./scripts/run_prompt_creation.py \
  --speaker_name "Jenny" \
  --is_single_speaker \
  --dataset_name "grigorimaxim/jenny-tts-tags-6h" \
  --output_dir "./tmp_jenny" \
  --dataset_config_name "default" \
  --model_name_or_path "google/gemma-2b-it" \
  --per_device_eval_batch_size 12 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 2 \
  --push_to_hub \
  --hub_dataset_id "jenny-tts-6h-tagged" \
  --preprocessing_num_workers 2