import argparse
import time

from datasets import load_from_disk

from src.data_preprocessing.ner_preprocess import NERPreprocessor
from src.fine_tuning.ner_finetuning import define_model, MetricsComputer, define_trainer
from src.utils.config_loader import load_ner_config
from src.utils.data_loader import create_hf_dataset_from_brats
from src.utils.ner_utils import extract_label_maps, load_label_maps

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification
)


def __get_args():
    # Manage arguments
    parser = argparse.ArgumentParser(description="Tokenization and data preparation for fine-tuning EriBERTa")
    parser.add_argument("--force_tokenize", action="store_true", help="Force to tokenize raw data")
    return parser.parse_args()


def run_ner_pipeline():
    # Load NER general configs
    config = load_ner_config("config.yaml")

    # Paths from configuration file
    TRAIN_RAW = config["paths"]["ner"]["raw"]["train"]
    DEV_RAW = config["paths"]["ner"]["raw"]["dev"]
    TRAIN_PROCESSED = config["paths"]["ner"]["processed"]["train"]
    DEV_PROCESSED = config["paths"]["ner"]["processed"]["dev"]
    LABEL2ID_PATH = config["paths"]["ner"]["label_map"]
    RESULTS_PATH = config["paths"]["ner"]["results"]

    # Model name and tokenizer
    model_name = "HiTZ/EriBERTa-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Arguments
    args = __get_args()

    if args.force_tokenize:

        # Load text+ann (BRAT) data into HF Dataset
        print("\n⏳ Loading raw data...")
        start_time = time.time()
        train_dataset = create_hf_dataset_from_brats(TRAIN_RAW)
        dev_dataset = create_hf_dataset_from_brats(DEV_RAW)
        print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")
        # TODO Prueba (borrar cuando se compruebe)
        print(train_dataset[0])

        # Label maps
        print("\n⏳ Extracting label maps...")
        start_time = time.time()
        label2id, id2label = extract_label_maps(train_dataset, LABEL2ID_PATH)
        print(f"✅ Label maps extracted in {time.time() - start_time:.2f} seconds.\n")

        # NERPreprocessor class
        preprocessor = NERPreprocessor(tokenizer, label2id, id2label)

        # Tokenize
        print("\n⏳ Tokenizing data...")
        start_time = time.time()
        train_tokenized = preprocessor.process_dataset(train_dataset)
        dev_tokenized = preprocessor.process_dataset(dev_dataset)
        # Save tokenized data to disk
        train_tokenized.save_to_disk(TRAIN_PROCESSED)
        dev_tokenized.save_to_disk(DEV_PROCESSED)
        print(f"✅ Data tokenized in {time.time() - start_time:.2f} seconds.\n")

    else:

        print("\n⏳ Loading processed data...")
        start_time = time.time()
        # Load processed data
        train_tokenized = load_from_disk(TRAIN_PROCESSED)
        dev_tokenized = load_from_disk(DEV_PROCESSED)
        # Load label maps
        label2id, id2label = load_label_maps(LABEL2ID_PATH)
        print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print("\n⏳ Starting fine-tuning process...")
    start_time = time.time()
    # Define model
    model = define_model(model_name, label2id, id2label)
    # Metrics computer
    metrics_computer = MetricsComputer(id2label)
    # Define trainer
    trainer = define_trainer(model, tokenizer, data_collator, train_tokenized, dev_tokenized,
                             metrics_computer.compute_metrics, RESULTS_PATH)
    # Fine-tuning
    trainer.train()
    print(f"✅ Model fine-tuned in {time.time() - start_time:.2f} seconds.\n")

    print("\n⏳ Evaluating fine-tuned model...")
    metrics = trainer.evaluate()
    print(metrics)
    print(f"✅ Model evaluated in {time.time() - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    run_ner_pipeline()
