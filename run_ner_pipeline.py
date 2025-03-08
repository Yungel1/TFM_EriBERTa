import argparse
import os
import time

import torch
from datasets import load_from_disk

from src.data_preprocessing.ner_preprocess import NERPreprocessor, flatten_dataset
from src.evaluation.ner_prediction import predict_and_save
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
    parser.add_argument("--force_fine_tuning", action="store_true", help="Force to fine-tune the model")
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

    if args.force_tokenize or not os.path.exists(TRAIN_PROCESSED) or not os.path.exists(DEV_PROCESSED):

        # Load text+ann (BRAT) data into HF Dataset
        print("\n⏳ Loading raw data...")
        start_time = time.time()
        train_dataset = create_hf_dataset_from_brats(TRAIN_RAW)
        dev_dataset = create_hf_dataset_from_brats(DEV_RAW)
        print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

        print("\n⏳ Extracting label maps...")
        start_time = time.time()
        label2id, id2label = extract_label_maps(train_dataset, LABEL2ID_PATH)
        print(f"✅ Label maps extracted in {time.time() - start_time:.2f} seconds.\n")

        # NERPreprocessor class
        preprocessor = NERPreprocessor(tokenizer, label2id, id2label)

        print("\n⏳ Tokenizing data...")
        start_time = time.time()
        train_tokenized = preprocessor.tokenize_dataset(train_dataset)
        dev_tokenized = preprocessor.tokenize_dataset(dev_dataset)
        # Save tokenized data to disk
        train_tokenized.save_to_disk(TRAIN_PROCESSED)
        dev_tokenized.save_to_disk(DEV_PROCESSED)
        print(f"✅ Data tokenized in {time.time() - start_time:.2f} seconds.\n")

        # TODO Borrar cuando se confirme que no hace falta
        # print("\n⏳ Flattening tokenized dataset...")
        # start_time = time.time()
        # train_flattened = flatten_dataset(train_tokenized)
        # dev_flattened = flatten_dataset(dev_tokenized)
        # # Save flattened data to disk
        # train_flattened.save_to_disk(TRAIN_PROCESSED)
        # dev_flattened.save_to_disk(DEV_PROCESSED)
        # print(f"✅ Dataset flattened in {time.time() - start_time:.2f} seconds.\n")

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

    # Metrics computer
    metrics_computer = MetricsComputer(id2label, True)

    BEST_MODEL_PATH = f"{RESULTS_PATH}/best_model"

    # GPU or CPU (GPU when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nℹ️ Model is using: {device}\n")

    if args.force_fine_tuning or not os.path.exists(BEST_MODEL_PATH):

        print("\n⏳ Starting fine-tuning process...")
        start_time = time.time()
        # Define model
        model = define_model(model_name, label2id, id2label)
        model.to(device)

        # Define trainer
        trainer = define_trainer(model, tokenizer, data_collator, train_tokenized, dev_tokenized,
                                 metrics_computer.compute_metrics, RESULTS_PATH)

        # Fine-tuning and saving best model
        train_result = trainer.train()
        metrics_train = train_result.metrics
        # trainer.log_metrics("train", metrics_train)
        trainer.save_metrics("train", metrics_train)
        if trainer.state.best_model_checkpoint:
            trainer.save_model(BEST_MODEL_PATH)
        print(f"✅ Model fine-tuned in {time.time() - start_time:.2f} seconds.\n")

    else:
        print("\n⏳ Loading best fine-tuned model...")
        start_time = time.time()
        # Define model
        model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_PATH)
        model.to(device)
        # model.push_to_hub("Yungel1/EriBERTa_NER")
        # model = AutoModelForTokenClassification.from_pretrained("Yungel1/EriBERTa_NER")
        # Define trainer
        trainer = define_trainer(model, tokenizer, data_collator, train_tokenized, dev_tokenized,
                                 metrics_computer.compute_metrics, RESULTS_PATH)
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds.\n")

    print("\n⏳ Evaluating fine-tuned model...")
    start_time = time.time()
    metrics_eval = trainer.evaluate()
    # trainer.log_metrics("eval", metrics_eval)
    trainer.save_metrics("eval", metrics_eval)
    print(f"✅ Model evaluated in {time.time() - start_time:.2f} seconds.\n")

    print("\n⏳ Starting inference and final evaluation...")
    start_time = time.time()
    predict_and_save(trainer, dev_tokenized, id2label, metrics_computer.compute_metrics, RESULTS_PATH)
    print(f"✅ Inference and save finished in {time.time() - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    run_ner_pipeline()
