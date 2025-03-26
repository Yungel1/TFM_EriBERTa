import argparse
import logging
import os
import time

from datasets import load_from_disk
from transformers import (
    set_seed
)

from harness.scripts.gemma_ner_preprocess import process_dataset
from src.utils.config_loader import load_config
from src.utils.data_loader import create_hf_dataset_from_json
from src.utils.ner_utils import extract_label_maps, load_label_maps

# Logs configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set seed
set_seed(42)


def __get_args():
    # Manage arguments
    parser = argparse.ArgumentParser(description="Gemma 3 inference and evaluation")
    parser.add_argument("--force_process", action="store_true", help="Force to process raw data")
    parser.add_argument("--config_path", type=str, default="config/ner/config_ner_cantemist.yaml", help="Config path")
    return parser.parse_args()


def run_ner_gemma_pipeline():
    # Arguments
    args = __get_args()

    # Load NER general configs
    config = load_config(args.config_path)

    # Paths from configuration file
    TEST_JSON = config["paths"]["json_raw"].get("test")
    TEST_PROCESSED = config["paths"]["gemma_processed"].get("test")
    LABEL2ID_PATH = config["paths"]["label_map"]

    # Model name and tokenizer
    model_name = "google/gemma-3-1b-it"

    if args.force_process or not os.path.exists(TEST_PROCESSED):
        # Load JSON data into HF Dataset
        logger.info(f"\n⏳ Loading raw data (test path: {TEST_JSON}) ...")
        start_time = time.time()
        test = create_hf_dataset_from_json(TEST_JSON)
        logger.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

        logger.info("\n⏳ Extracting label maps...")
        start_time = time.time()
        label2id, id2label = extract_label_maps(test, LABEL2ID_PATH)
        logger.info(f"✅ Label maps extracted in {time.time() - start_time:.2f} seconds.\n")

        logger.info("\n⏳ Tokenizing data...")
        start_time = time.time()
        test_processed = process_dataset(test)
        # Save tokenized data to disk
        test_processed.save_to_disk(TEST_PROCESSED)
        logger.info(f"✅ Data processed in {time.time() - start_time:.2f} seconds.\n")

    else:
        logger.info(f"\n⏳ Loading processed data (train path: {TEST_PROCESSED}) ...")
        start_time = time.time()
        # Load processed data
        test_tokenized = load_from_disk(TEST_PROCESSED)
        # Load label maps
        label2id, id2label = load_label_maps(LABEL2ID_PATH)
        logger.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    run_ner_gemma_pipeline()
