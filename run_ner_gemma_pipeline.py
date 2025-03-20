import argparse
import logging
import re
import json
import time

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import evaluate

from src.utils.config_loader import load_config
from src.utils.data_loader import create_hf_dataset_from_brats
from src.utils.ner_utils import extract_label_maps


def __get_args():
    # Manage arguments
    parser = argparse.ArgumentParser(description="Gemma 3 inference and evaluation")
    parser.add_argument("--config_path", type=str, default="config/config_casimedicos_ner.yaml", help="Config path")
    return parser.parse_args()


def extract_entities(text, label2id):
    # Simulated function to extract entities from generated text
    # You need to replace this with actual named entity recognition logic
    entities = []
    words = text.split()
    for i, word in enumerate(words):
        if word in label2id:
            entities.append({"label": label2id[word], "start": i, "end": i + len(word)})
    return entities


def infer_and_evaluate(dataset, model, tokenizer, label2id, id2label):
    model.eval()
    predictions = []
    references = []

    for instance in dataset:
        text = instance["text"]
        true_entities = instance["entities"]

        # Construcción del prompt para zero-shot NER
        prompt = (f"Extrae solo las entidades nombradas que sean Claim o Premise del siguiente texto:"
                  f"\n{text}\n\nImprime como salida las entidades nombradas en formato JSON.")

        # Tokenizar el prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Procesar la salida del modelo para extraer entidades
        predicted_entities = extract_entities(generated_text, label2id)

        # Convertir a formato seqeval
        predictions.append([id2label[ent["label"]] for ent in predicted_entities])
        references.append([id2label[ent["label"]] for ent in true_entities])

    # Evaluar con seqeval
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=references)

    return results


def run_ner_gemma_pipeline():
    # Arguments
    args = __get_args()

    # Logs configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load NER general configs
    config = load_config(args.config_path)

    # Paths from configuration file
    TEST_RAW = config["paths"]["raw"].get("test") or config["paths"]["raw"]["dev"]
    LABEL2ID_PATH = config["paths"]["label_map"]

    # Load text+ann (BRAT) data into HF Dataset
    logging.info(f"\n⏳ Loading raw data (test path: {TEST_RAW}) ...")
    start_time = time.time()
    dataset = create_hf_dataset_from_brats(TEST_RAW)
    logging.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")

    logging.info("\n⏳ Extracting label maps...")
    start_time = time.time()
    label2id, id2label = extract_label_maps(dataset, LABEL2ID_PATH)
    logging.info(f"✅ Label maps extracted in {time.time() - start_time:.2f} seconds.\n")

    # Model and tokenizer
    model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    results = infer_and_evaluate(dataset, model, tokenizer, label2id, id2label)

    print(results)


if __name__ == "__main__":
    run_ner_gemma_pipeline()
