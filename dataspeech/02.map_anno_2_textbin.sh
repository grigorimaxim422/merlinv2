python ./scripts/metadata_to_text.py \
    "grigorimaxim/jenny-tts-tags-6h" \
    --repo_id "jenny-tts-tags-6h" \
    --configuration "default" \
    --cpu_num_workers 2 \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation