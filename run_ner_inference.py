import argparse
import logging
import os

import torch

from src.data_preprocessing.ner_preprocess import NERPreprocessor
from src.evaluation.ner_prediction import predict_and_save
from src.fine_tuning.ner_finetuning import MetricsComputer
from src.utils.data_loader import create_hf_dataset_from_brats

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    set_seed, Trainer,
)


os.environ["WANDB_DISABLED"] = "true"

# Logs configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set seed
set_seed(42)

VERY_LARGE_INTEGER = int(1e30)


def main():
    os.environ["WANDB_DISABLED"] = "true"  # <- desactiva wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text_file_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.model_max_length == VERY_LARGE_INTEGER:
        tokenizer.model_max_length = 512

    # GPU or CPU (GPU when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load text+ann (BRAT) data into HF Dataset
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, device_map=device)
    txt_dataset = create_hf_dataset_from_brats(args.text_file_dir)
    label_list = model.config.id2label
    id2label = {int(k): v for k, v in label_list.items()}
    label2id = {v: k for k, v in id2label.items()}

    # NERPreprocessor class
    preprocessor = NERPreprocessor(tokenizer, label2id, id2label)

    txt_tokenized = preprocessor.tokenize_dataset(txt_dataset)

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # Metrics computer
    metrics_computer = MetricsComputer(id2label, True)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_computer.compute_metrics,
    )

    predict_and_save(trainer, txt_tokenized, id2label, metrics_computer.compute_metrics,
                     tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
