python ./scripts/metadata_to_text.py \
    "_cache_tags" \
    --configuration "default" \
    --cpu_num_workers 2 \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation \
    --output_dir "_cache_tags_bin"
#    --repo_id "jenny-tts-tags-6h" \
    