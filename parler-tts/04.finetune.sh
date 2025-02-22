 accelerate launch  -m --multi_gpu training.run_parler_tts_training \
    --model_name_or_path "parler-tts/parler-tts-mini-v1" \
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
    --output_dir "../output_dir_training/" \
    --temporary_save_to_disk "../audio_code_tmp/" \
    --save_to_disk "../tmp_dataset_audio/" \
    --dataloader_num_workers 2 \
    --do_eval \
    --predict_with_generate \
    --include_inputs_for_metrics \
    --group_by_length true