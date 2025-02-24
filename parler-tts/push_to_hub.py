from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
    build_delay_pattern_mask,
)
DATA_CACHE_DIR="../_cache/"
from huggingface_hub import HfApi
import argparse
import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry

if __name__ == "__main__":                
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--model_name_or_path", type=str, help="Path or name of the dataset. See: https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/loading_methods#datasets.load_dataset.path")
        parser.add_argument("--cache_dir", default=None, type=str, help="Dataset configuration to use, if necessary.")        
        parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")        
        args = parser.parse_args()    

        model = ParlerTTSForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                cache_dir=(args.cache_dir or DATA_CACHE_DIR))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                cache_dir=(args.cache_dir or DATA_CACHE_DIR))
        # tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini-v1",
        #                                         cache_dir=(args.cache_dir or DATA_CACHE_DIR))
        print(f"Loading model and tokenizer done!")

        model.push_to_hub(args.repo_id)
        tokenizer.push_to_hub(args.repo_id)
        print(f"Saving model {args.repo_id} done!")

