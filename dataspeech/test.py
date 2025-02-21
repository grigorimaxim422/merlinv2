from datasets import load_dataset
from utils import save_dataset02,save_dataset

dataset = load_dataset('ylacombe/jenny-tts-6h')
print(dataset['train'][0])
save_dataset(dataset['train'], '_tmp_cache_02')