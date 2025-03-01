import time

from src.data_preprocessing.ner_preprocess import NERPreprocessor
from src.utils.config_loader import load_ner_config

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

# Load NER general configs
config = load_ner_config("config.yaml")

# Paths from configuration file
TRAIN_RAW = config["paths"]["ner"]["raw"]["train"]
DEV_RAW = config["paths"]["ner"]["raw"]["dev"]
TRAIN_PROCESSED = config["paths"]["ner"]["processed"]["train"]
DEV_PROCESSED = config["paths"]["ner"]["processed"]["dev"]

# Model to be trained
medical_model = "HiTZ/EriBERTa-base"
tokenizer = AutoTokenizer.from_pretrained(medical_model)

# NERPreprocessor class
preprocessor = NERPreprocessor(tokenizer)

# Load text+ann (BRAT) data into HF Dataset
print("\n⏳ Loading data...")
start_time = time.time()
train_dataset = preprocessor.create_hf_dataset_from_brats(TRAIN_RAW)
dev_dataset = preprocessor.create_hf_dataset_from_brats(DEV_RAW)
print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")
# TODO Prueba (borrar cuando se compruebe)
print(train_dataset[0])

# Save HF Dataset to disk
train_dataset.save_to_disk(TRAIN_PROCESSED)
dev_dataset.save_to_disk(DEV_PROCESSED)

# Tokenize
print("\n⏳ Tokenizing data...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(medical_model)
train_tokenized = preprocessor.process_dataset(train_dataset)
dev_tokenized = preprocessor.process_dataset(dev_dataset)
print(f"✅ Data tokenized in {time.time() - start_time:.2f} seconds.\n")

sample = train_tokenized[0]
print("\nSample processed item:")
for key, value in sample.items():
    print(f"{key}: {value[:50]}...")

texto_original = train_dataset[0]["text"]
print(preprocessor.label2id)
for i, (offset, label) in enumerate(zip(sample["offset_mapping"][0], sample["labels"][0])):
    inicio, fin = offset
    fragmento = texto_original[inicio:fin]
    print(f"Token {i}: '{fragmento}' -> Label: {label}")

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
