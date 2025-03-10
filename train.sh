cd dataspeech
python3 main.py "ylacombe/jenny-tts-6h" \
  --configuration "default" \
  --text_column_name "transcription" \
  --audio_column_name "audio" \
  --cpu_num_workers 2 \
  --num_workers_per_gpu_for_pitch 2 \
  --output_dir "_cache_tags" \
  --rename_column

python ./scripts/metadata_to_text.py \
    "_cache_tags" \
    --configuration "default" \
    --cpu_num_workers 2 \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation \
    --output_dir "_cache_tags_bin"
#    --repo_id "jenny-tts-tags-6h" \

python ./scripts/run_prompt_creation.py \
  --speaker_name "Jenny" \
  --is_single_speaker \
  --dataset_name "_cache_tags_bin" \
  --output_dir "../_cache_datasets" \
  --dataset_config_name "default" \
  --model_name_or_path "google/gemma-2b-it" \
  --cache_dir "../_cache" \
  --per_device_eval_batch_size 12 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 2 \
  --preprocessing_num_workers 2

cd ..
cd parler-tts

 accelerate launch  -m --multi_gpu training.run_parler_tts_training \
    --model_name_or_path "Legalaz/llabo_01_13_08_25" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "parler-tts/parler-tts-mini-v1" \
    --prompt_tokenizer_name "parler-tts/parler-tts-mini-v1" \
    --overwrite_output_dir true \
    --train_dataset_name "ylacombe/jenny-tts-6h" \
    --train_metadata_dataset_name "../_cache_datasets" \
    --train_dataset_config_name "default" \
    --train_split_name "train" \
    --eval_dataset_name "ylacombe/jenny-tts-6h" \
    --eval_metadata_dataset_name "../_cache_datasets" \
    --eval_dataset_config_name "default" \
    --cache_dir "../_cache" \
    --eval_split_name "train" \
    --max_eval_samples 8 \
    --per_device_eval_batch_size 8 \
    --target_audio_column_name "audio" \
    --description_column_name "text_description" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 20 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 400 \
    --preprocessing_num_workers 2 \
    --do_train true \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 18 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 2 \
    --learning_rate 0.00008 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 50 \
    --logging_steps 2 \
    --freeze_text_encoder true \
    --audio_encoder_per_device_batch_size 4 \
    --dtype "float16" \
    --seed 456 \
    --output_dir "./r310_lab_jen6/" \
    --temporary_save_to_disk "../audio_code_tmp/" \
    --save_to_disk "../tmp_dataset_audio/" \
    --dataloader_num_workers 2 \
    --do_eval \
    --predict_with_generate \
    --include_inputs_for_metrics \
    --group_by_length true

rm -rf parler-speech/r310_lab_jen6
rm -rf    parler-speech/*checkpoint*

# cd ..

python3 push_to_hub.py \
    --model_name_or_path "./r310_lab_jen6" \
    --cache_dir "../_cache" \
    --repo_id "r310_lab_jen6"
