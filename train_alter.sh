cd parler-tts
accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "godofmining/shidou14" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "parler-tts/parler-tts-mini-v1" \
    --prompt_tokenizer_name "parler-tts/parler-tts-mini-v1" \
    --report_to "wandb" \
    --overwrite_output_dir true \
    --train_dataset_name "parler-tts/libritts_r_filtered" \
    --train_metadata_dataset_name "parler-tts/libritts-r-filtered-speaker-descriptions" \
    --train_dataset_config_name "clean" \
    --train_split_name "train.clean.100" \
    --eval_dataset_name "parler-tts/libritts_r_filtered" \
    --eval_metadata_dataset_name "parler-tts/libritts-r-filtered-speaker-descriptions" \
    --eval_dataset_config_name "other" \
    --eval_split_name "test.other" \
    --target_audio_column_name "audio" \
    --description_column_name "text_normalized" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 600 \
    --add_audio_samples_to_wandb true \
    --id_column_name "id" \
    --preprocessing_num_workers 8 \
    --do_train true \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing false \
    --per_device_train_batch_size 4 \
    --learning_rate 0.00095 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 20000 \
    --logging_steps 1000 \
    --freeze_text_encoder true \
    --do_eval true \
    --predict_with_generate true \
    --include_inputs_for_metrics true \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_steps 10000 \
    --per_device_eval_batch_size 4 \
    --audio_encoder_per_device_batch_size 24 \
    --dtype "bfloat16" \
    --seed 456 \
    --output_dir "./r225_go14_lib1/" \
    --temporary_save_to_disk "../audio_code_tmp/" \
    --save_to_disk "../tmp_dataset_audio/" \
    --max_eval_samples 96 \
    --dataloader_num_workers 8 \
    --group_by_length true \
    --attn_implementation "sdpa"


python3 push_to_hub.py \
    --model_name_or_path "./r225_go14_lib1" \
    --cache_dir "../_cache" \
    --repo_id "r225_go14_lib1"
