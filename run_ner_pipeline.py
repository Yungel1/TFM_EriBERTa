import argparse
import logging
import os
import time

import torch
import wandb
from datasets import load_from_disk

from src.data_preprocessing.ner_preprocess import NERPreprocessor
from src.evaluation.ner_prediction import predict_and_save
from src.fine_tuning.ner_finetuning import define_config, MetricsComputer, define_trainer
from src.utils.config_loader import load_config
from src.utils.data_loader import create_hf_dataset_from_brats
from src.utils.ner_utils import extract_label_maps, load_label_maps

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification
)

from src.fine_tuning.wandb_tuning import configure_sweep, train_model_wandb


def __get_args():
    # Manage arguments
    parser = argparse.ArgumentParser(description="Tokenization and data preparation for fine-tuning EriBERTa")
    parser.add_argument("--force_tokenize", action="store_true", help="Force to tokenize raw data")
    parser.add_argument("--force_fine_tuning", action="store_true", help="Force to fine-tune the model")
    parser.add_argument("--opt_hyperparameters", action="store_true", help="Execute hyperparameter optimization process")
    parser.add_argument("--config_path", type=str, default="config/config_casimedicos_ner.yaml", help="Config path")
    return parser.parse_args()


def run_ner_pipeline():
    # Arguments
    args = __get_args()

    # Logs configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load NER general configs
    config = load_config(args.config_path)

    # Paths from configuration file
    TRAIN_RAW = config["paths"]["raw"]["train"]
    DEV_RAW = config["paths"]["raw"]["dev"]
    TEST_RAW = config["paths"]["raw"].get("test") or DEV_RAW
    TRAIN_PROCESSED = config["paths"]["processed"]["train"]
    DEV_PROCESSED = config["paths"]["processed"]["dev"]
    TEST_PROCESSED = config["paths"]["processed"].get("test") or DEV_PROCESSED
    LABEL2ID_PATH = config["paths"]["label_map"]
    RESULTS_PATH = config["paths"]["results"]

    # Model name and tokenizer
    model_name = "HiTZ/EriBERTa-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.force_tokenize or not os.path.exists(TRAIN_PROCESSED) or not os.path.exists(DEV_PROCESSED):

        # Load text+ann (BRAT) data into HF Dataset
        logging.info(f"\n⏳ Loading raw data (train path: {TRAIN_PROCESSED}) ...")
        start_time = time.time()
        train_dataset = create_hf_dataset_from_brats(TRAIN_RAW)
        dev_dataset = create_hf_dataset_from_brats(DEV_RAW)
        test_dataset = create_hf_dataset_from_brats(TEST_RAW)
        logging.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

        logging.info("\n⏳ Extracting label maps...")
        start_time = time.time()
        label2id, id2label = extract_label_maps(train_dataset, LABEL2ID_PATH)
        logging.info(f"✅ Label maps extracted in {time.time() - start_time:.2f} seconds.\n")

        # NERPreprocessor class
        preprocessor = NERPreprocessor(tokenizer, label2id, id2label)

        logging.info("\n⏳ Tokenizing data...")
        start_time = time.time()
        train_tokenized = preprocessor.tokenize_dataset(train_dataset)
        dev_tokenized = preprocessor.tokenize_dataset(dev_dataset)
        test_tokenized = preprocessor.tokenize_dataset(test_dataset)
        # Save tokenized data to disk
        train_tokenized.save_to_disk(TRAIN_PROCESSED)
        dev_tokenized.save_to_disk(DEV_PROCESSED)
        test_tokenized.save_to_disk(TEST_PROCESSED)
        logging.info(f"✅ Data tokenized in {time.time() - start_time:.2f} seconds.\n")

    else:
        logging.info(f"\n⏳ Loading processed data (train path: {TRAIN_PROCESSED}) ...")
        start_time = time.time()
        # Load processed data
        train_tokenized = load_from_disk(TRAIN_PROCESSED)
        dev_tokenized = load_from_disk(DEV_PROCESSED)
        test_tokenized = load_from_disk(TEST_PROCESSED)
        # Load label maps
        label2id, id2label = load_label_maps(LABEL2ID_PATH)
        logging.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # Metrics computer
    metrics_computer = MetricsComputer(id2label, True)

    BEST_MODEL_PATH = f"{RESULTS_PATH}/best_model"

    # GPU or CPU (GPU when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nℹ️ Model is using: {device}\n")

    # Hyperparameter optimization
    if args.opt_hyperparameters:
        logging.info("\n⏳ Starting hyperparameter optimization process...")
        start_time = time.time()
        # Define model config
        config = define_config(model_name, label2id, id2label)
        # Wandb
        sweep_id = configure_sweep()
        wandb.agent(
            sweep_id,
            function=lambda: train_model_wandb(model_name, config, device, tokenizer, data_collator,
                                               train_tokenized, dev_tokenized, metrics_computer.compute_metrics,
                                               RESULTS_PATH),
            count=10
        )
        logging.info(f"✅ Hyperparameter optimization process done in {time.time() - start_time:.2f} seconds.\n")
        return

    # Hyperparameters from config
    hyperparameters = {
        "batch_size": config["hyperparameters"]["batch_size"],
        "learning_rate": config["hyperparameters"]["learning_rate"],
        "weight_decay": config["hyperparameters"]["weight_decay"],
    }

    if args.force_fine_tuning or not os.path.exists(BEST_MODEL_PATH):

        logging.info("\n⏳ Starting fine-tuning process...")
        start_time = time.time()
        # Define model
        config = define_config(model_name, label2id, id2label)
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.to(device)

        # Define trainer
        trainer = define_trainer(model, hyperparameters, tokenizer, data_collator, train_tokenized, dev_tokenized,
                                 metrics_computer.compute_metrics, RESULTS_PATH)

        # Fine-tuning and saving best model
        train_result = trainer.train()
        metrics_train = train_result.metrics
        # trainer.log_metrics("train", metrics_train)
        trainer.save_metrics("train", metrics_train)
        if trainer.state.best_model_checkpoint:
            trainer.save_model(BEST_MODEL_PATH)
        logging.info(f"✅ Model fine-tuned in {time.time() - start_time:.2f} seconds.\n")

    else:
        logging.info("\n⏳ Loading best fine-tuned model...")
        start_time = time.time()
        # Define model
        model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_PATH)
        model.to(device)
        # model.push_to_hub("Yungel1/EriBERTa_NER")
        # model = AutoModelForTokenClassification.from_pretrained("Yungel1/EriBERTa_NER")
        # Define trainer
        trainer = define_trainer(model, hyperparameters, tokenizer, data_collator, train_tokenized, dev_tokenized,
                                 metrics_computer.compute_metrics, RESULTS_PATH)
        logging.info(f"✅ Model loaded in {time.time() - start_time:.2f} seconds.\n")

    logging.info("\n⏳ Evaluating fine-tuned model...")
    start_time = time.time()
    metrics_eval = trainer.evaluate()
    # trainer.log_metrics("eval", metrics_eval)
    trainer.save_metrics("eval", metrics_eval)
    logging.info(f"✅ Model evaluated in {time.time() - start_time:.2f} seconds.\n")

    logging.info("\n⏳ Starting inference and final evaluation...")
    start_time = time.time()
    predict_and_save(trainer, test_tokenized, id2label, metrics_computer.compute_metrics, tokenizer, RESULTS_PATH)
    logging.info(f"✅ Inference and save finished in {time.time() - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    run_ner_pipeline()
