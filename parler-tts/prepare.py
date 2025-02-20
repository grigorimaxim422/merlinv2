import logging
import os
import re
import sys
import time
import math
import contextlib
from multiprocess import set_start_method
from datetime import timedelta
import inspect
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry


from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, DistributedDataParallelKwargs
from accelerate.utils.memory import release_memory

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
    build_delay_pattern_mask,
)

from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from training.arguments import ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments
from training.data import load_multiple_datasets, DataCollatorParlerTTSWithPadding, DataCollatorEncodecWithPadding
from training.eval import clap_similarity, wer, si_sdr

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    if data_args.pad_to_max_length and (
        data_args.max_duration_in_seconds is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "`pad_to_max_length` is `True` but one of the following parameters has not been set: `max_duration_in_seconds`, `max_prompt_token_length`, `max_description_token_length`"
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    ####### A. Preparation
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120)), DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir)
    del feature_extractor
    
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir)
    del prompt_tokenizer
    
    description_tokenizer = AutoTokenizer.from_pretrained(
        model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir)
    del description_tokenizer
    
    raw_datasets["train"] = load_multiple_datasets(
        accelerator,
        data_args.train_dataset_name,
        data_args.train_dataset_config_name,
        metadata_dataset_names=data_args.train_metadata_dataset_name,
        splits=data_args.train_split_name,
        dataset_samples=data_args.train_dataset_samples,
        seed=training_args.seed,
        cache_dir=model_args.cache_dir,
        num_proc=data_args.preprocessing_num_workers,
        id_column_name=data_args.id_column_name,
        columns_to_keep=columns_to_keep.values(),
        prompt_column_name=data_args.prompt_column_name,
        audio_column_name=data_args.target_audio_column_name,
        sampling_rate=sampling_rate,
        logger=logger,
        # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
    )
    del raw_datasets['train']
    
    raw_datasets["eval"] = load_multiple_datasets(
        accelerator,
        data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
        data_args.eval_dataset_config_name
        if data_args.eval_dataset_config_name
        else data_args.train_dataset_config_name,
        metadata_dataset_names=data_args.eval_metadata_dataset_name,
        splits=data_args.eval_split_name,
        cache_dir=model_args.cache_dir,
        num_proc=data_args.preprocessing_num_workers,
        id_column_name=data_args.id_column_name,
        columns_to_keep=columns_to_keep.values(),
        prompt_column_name=data_args.prompt_column_name,
        audio_column_name=data_args.target_audio_column_name,
        sampling_rate=sampling_rate,
        logger=logger,
        # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
    )
    del raw_datasets['eval']
    
    config = ParlerTTSConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    del config
    
    print(f"All data loaded seems to cached in cache_dir {model_args.cache_dir}")
    

if __name__ == "__main__":
    main()
