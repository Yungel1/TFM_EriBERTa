import argparse
import time

from datasets import load_from_disk

from src.data_preprocessing.ner_preprocess import NERPreprocessor
from src.utils.config_loader import load_ner_config
from src.utils.data_loader import create_hf_dataset_from_brats
from src.utils.ner_utils import extract_label_maps, load_label_maps

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
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

    # Model to be trained
    medical_model = "HiTZ/EriBERTa-base"
    tokenizer = AutoTokenizer.from_pretrained(medical_model)

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
        # TODO Prueba (borrar cuando se compruebe)
        print(train_tokenized[0])
        print(label2id)


    # # Fine-tuning del modelo
    # print("\n Iniciando fine-tuning de EriBERTa...")
    # start_time = time.time()
    # fine_tune_model(train_data, dev_data, MODEL_DIR, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    # print(f"✅ Fine-tuning completado en {time.time() - start_time:.2f} segundos.")
    #
    # # Evaluación del modelo
    # print("\n Evaluando el modelo...")
    # metrics = evaluate_model(MODEL_DIR, test_data)
    # print("Resultados de evaluación:", metrics)
    #
    # print("\n ¡Pipeline completado con éxito!")


if __name__ == "__main__":
    run_ner_pipeline()
