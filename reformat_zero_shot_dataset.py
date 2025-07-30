import os
import logging
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reformat_zero_shot")


def reformat_prompt_and_labels(sample):
    input_ids = sample['input_ids']
    labels = sample['labels']
    prompt_len = len(input_ids)
    trimmed_labels = labels[prompt_len:]
    sample['labels'] = trimmed_labels
    return sample

def process_and_save_datasets(input_dirs, output_dir, tokenizer_name="v1kram/zer_v3"):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for input_dir in input_dirs:
        logger.info(f"Loading dataset from {input_dir}")
        ds = load_from_disk(input_dir)
        logger.info(f"Dataset loaded with {len(ds)} samples. Columns: {ds.column_names}")
        logger.info(f"Reformatting and trimming labels for {input_dir}")
        ds = ds.map(reformat_prompt_and_labels)
        base_name = os.path.basename(os.path.normpath(input_dir))
        out_path = os.path.join(output_dir, base_name)
        logger.info(f"Saving reformatted dataset to {out_path}")
        ds.save_to_disk(out_path)
        logger.info(f"Saved {len(ds)} samples to {out_path}")
        # Log a few samples for verification
        for i in range(min(3, len(ds))):
            logger.info(f"Sample {i} input_ids[:10]: {ds[i]['input_ids'][:10]}")
            logger.info(f"Sample {i} trimmed labels: {ds[i]['labels']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch reformat and save zero-shot datasets with trimmed labels.")
    parser.add_argument('--input-dirs', nargs='+', required=True, help='List of dataset directories to process')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save reformatted datasets')
    args = parser.parse_args()
    process_and_save_datasets(args.input_dirs, args.output_dir)


