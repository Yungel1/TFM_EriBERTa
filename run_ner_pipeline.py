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
    DataCollatorForTokenClassification,
    set_seed,
)

from src.fine_tuning.wandb_tuning import configure_sweep, train_model_wandb

# Logs configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set seed
set_seed(42)


def __get_args():
    # Manage arguments
    parser = argparse.ArgumentParser(description="Tokenization and data preparation for fine-tuning EriBERTa")
    parser.add_argument("--model", type=str, default="eriberta", choices=["eriberta"], help="Model to use")
    parser.add_argument("--force_tokenize", action="store_true", help="Force to tokenize raw data")
    parser.add_argument("--force_fine_tuning", action="store_true", help="Force to fine-tune the model")
    parser.add_argument("--opt_hyperparameters", action="store_true", help="Execute hyperparameter optimization process")
    parser.add_argument("--config_path", type=str, default="config/config_casimedicos_ner.yaml", help="Config path")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to execute the fine-tuning process")
    return parser.parse_args()


def run_ner_pipeline():
    # Arguments
    args = __get_args()

    # Load NER general configs
    config = load_config(args.config_path)

    # Selected arg model
    arg_model = args.model

    # Paths from configuration file
    TRAIN_RAW = config["paths"]["raw"]["train"]
    DEV_RAW = config["paths"]["raw"]["dev"]
    TEST_RAW = config["paths"]["raw"].get("test") or DEV_RAW
    TRAIN_PROCESSED = config["paths"]["processed"]["train"]
    DEV_PROCESSED = config["paths"]["processed"]["dev"]
    TEST_PROCESSED = config["paths"]["processed"].get("test") or DEV_PROCESSED
    LABEL2ID_PATH = config["paths"]["label_map"]
    RESULTS_PATH = config["models"][arg_model]["results"]
    WANDB_PROJECT = config["models"][arg_model]["wandb_project"]
    MODEL_NAME = config["models"][arg_model]["model_name"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if args.force_tokenize or not os.path.exists(TRAIN_PROCESSED) or not os.path.exists(DEV_PROCESSED):

        # Load text+ann (BRAT) data into HF Dataset
        logger.info(f"\n⏳ Loading raw data (train path: {TRAIN_RAW}) ...")
        start_time = time.time()
        train_dataset = create_hf_dataset_from_brats(TRAIN_RAW)
        dev_dataset = create_hf_dataset_from_brats(DEV_RAW)
        test_dataset = create_hf_dataset_from_brats(TEST_RAW)
        logger.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

        logger.info("\n⏳ Extracting label maps...")
        start_time = time.time()
        label2id, id2label = extract_label_maps(train_dataset, LABEL2ID_PATH)
        logger.info(f"✅ Label maps extracted in {time.time() - start_time:.2f} seconds.\n")

        # NERPreprocessor class
        preprocessor = NERPreprocessor(tokenizer, label2id, id2label)

        logger.info("\n⏳ Tokenizing data...")
        start_time = time.time()
        train_tokenized = preprocessor.tokenize_dataset(train_dataset)
        dev_tokenized = preprocessor.tokenize_dataset(dev_dataset)
        test_tokenized = preprocessor.tokenize_dataset(test_dataset)
        # Save tokenized data to disk
        train_tokenized.save_to_disk(TRAIN_PROCESSED)
        dev_tokenized.save_to_disk(DEV_PROCESSED)
        test_tokenized.save_to_disk(TEST_PROCESSED)
        logger.info(f"✅ Data tokenized in {time.time() - start_time:.2f} seconds.\n")

    else:
        logger.info(f"\n⏳ Loading processed data (train path: {TRAIN_PROCESSED}) ...")
        start_time = time.time()
        # Load processed data
        train_tokenized = load_from_disk(TRAIN_PROCESSED)
        dev_tokenized = load_from_disk(DEV_PROCESSED)
        test_tokenized = load_from_disk(TEST_PROCESSED)
        # Load label maps
        label2id, id2label = load_label_maps(LABEL2ID_PATH)
        logger.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # Metrics computer
    metrics_computer = MetricsComputer(id2label, True)

    # GPU or CPU (GPU when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nℹ️ Model is using: {device}\n")

    # Hyperparameter optimization
    if args.opt_hyperparameters:

        logger.info("\n⏳ Starting hyperparameter optimization process...")
        start_time = time.time()
        # Define model config
        config = define_config(MODEL_NAME, label2id, id2label)
        # Wandb
        sweep_id = configure_sweep(WANDB_PROJECT)
        wandb.agent(
            sweep_id,
            function=lambda: train_model_wandb(MODEL_NAME, config, tokenizer, data_collator,
                                               train_tokenized, dev_tokenized, metrics_computer.compute_metrics,
                                               RESULTS_PATH),
            count=20
        )
        logger.info(f"✅ Hyperparameter optimization process done in {time.time() - start_time:.2f} seconds.\n")
        return

    # Hyperparameters from config
    hyperparameters = {
        "batch_size": config["models"][arg_model]["hyperparameters"]["batch_size"],
        "learning_rate": config["models"][arg_model]["hyperparameters"]["learning_rate"],
        "weight_decay": config["models"][arg_model]["hyperparameters"]["weight_decay"],
    }

    num_runs = args.runs - 1
    for run in range(0, args.runs):

        logger.info(f"\n⏳ Starting fine-tuning process {run}/{num_runs}...")
        start_time = time.time()

        RUN_RESULTS_PATH = f"{RESULTS_PATH}/run_{run}"
        BEST_MODEL_PATH = f"{RUN_RESULTS_PATH}/best_model"

        # Seed
        new_seed = 42 + run * 100
        set_seed(new_seed)
        logger.info(f"️ℹ️ Seed set to {new_seed}")

        if args.force_fine_tuning or not os.path.exists(BEST_MODEL_PATH):

            # Define model
            config = define_config(MODEL_NAME, label2id, id2label)
            model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True, device_map="auto")
            # model.to(device)

            # Define trainer
            trainer = define_trainer(model, hyperparameters, tokenizer, data_collator, train_tokenized, dev_tokenized,
                                     metrics_computer.compute_metrics, RUN_RESULTS_PATH)

            # Fine-tuning and saving best model
            train_result = trainer.train()
            metrics_train = train_result.metrics
            # trainer.log_metrics("train", metrics_train)
            trainer.save_metrics("train", metrics_train)
            if trainer.state.best_model_checkpoint:
                trainer.save_model(BEST_MODEL_PATH)
            logger.info(f"✅ Model fine-tuned in {time.time() - start_time:.2f} seconds. Saved to {BEST_MODEL_PATH}\n")

        else:
            logger.info("\n⏳ Loading best fine-tuned model...")
            start_time = time.time()
            # Define model
            model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_PATH, device_map='auto')
            # model.push_to_hub("Yungel1/EriBERTa_NER")
            # model = AutoModelForTokenClassification.from_pretrained("Yungel1/EriBERTa_NER")
            # Define trainer
            trainer = define_trainer(model, hyperparameters, tokenizer, data_collator, train_tokenized, dev_tokenized,
                                     metrics_computer.compute_metrics, RESULTS_PATH)
            logger.info(f"✅ Model loaded in {time.time() - start_time:.2f} seconds.\n")

        logger.info("\n⏳ Evaluating fine-tuned model...")
        start_time = time.time()
        metrics_eval = trainer.evaluate()
        # trainer.log_metrics("eval", metrics_eval)
        trainer.save_metrics("eval", metrics_eval)
        logger.info(f"✅ Model evaluated in {time.time() - start_time:.2f} seconds.\n")

        logger.info("\n⏳ Starting inference and final evaluation...")
        start_time = time.time()
        predict_and_save(trainer, test_tokenized, id2label, metrics_computer.compute_metrics, tokenizer, RESULTS_PATH)
        logger.info(f"✅ Inference and save finished in {time.time() - start_time:.2f} seconds.\n")

        with open(f"{RUN_RESULTS_PATH}/seed.txt", "w") as archivo:
            archivo.write(str(new_seed))
        logger.info(f"️ℹ️ Seed saved in {RUN_RESULTS_PATH}/seed.txt")

        del model, trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_ner_pipeline()
