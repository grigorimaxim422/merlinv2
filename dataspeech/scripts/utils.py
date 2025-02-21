import os
import shutil
from datasets import load_dataset
from datasets.features.features import require_decoding
from datasets.download.streaming_download_manager import xgetsize
from datasets import config
from datasets.utils.py_utils import convert_file_size_to_int
from datasets.table import embed_table_storage
from tqdm import tqdm
import pyarrow.parquet as pq

_max_shard_size = '400MB'#4e9
# data_dir = 'lj_speech_parquets'
# split = 'train'

def save_dataset(dataset, data_dir, split='train'):
    dataset_nbytes = dataset._estimate_nbytes()
    max_shard_size = convert_file_size_to_int(_max_shard_size)
    num_shards = int(dataset_nbytes / max_shard_size) + 1
    num_shards = max(num_shards, 1)
    #num_shards = 1
    shards = (dataset.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards))
    def shards_with_embedded_external_files(shards):
        for shard in shards:
            format = shard.format
            shard = shard.with_format("arrow")
            shard = shard.map(
                embed_table_storage,
                batched=True,
                batch_size=1000,
                keep_in_memory=True,
            )
            shard = shard.with_format(**format)
            yield shard
                        
    shards = shards_with_embedded_external_files(shards)
    
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        # Remove the directory
        shutil.rmtree(data_dir)
        print(f"Directory {data_dir} removed.")
    else:
        print(f"Directory {data_dir} does not exist.")
    os.makedirs(data_dir)

    for index, shard in tqdm(
        enumerate(shards),
        desc="Save the dataset shards",
        total=num_shards,
    ):
        shard_path = f"{data_dir}/{split}-{index:03d}-of-{num_shards:03d}.parquet"
        shard.to_parquet(shard_path)

def save_dataset02(dataset, data_dir):
    pq.write_table(dataset, f"{data_dir}/train-001.parquet")