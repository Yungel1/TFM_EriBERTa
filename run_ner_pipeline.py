import os
import time
from src.data_preprocessing import ner_preprocess
from src.utils.data_loader import load_txt_ann
from src.utils.config_loader import load_ner_config

# Load NER general configs
config = load_ner_config("config.yaml")

# Paths from configuration file
TRAIN_TEXT_RAW = config["paths"]["ner"]["raw"]["train"]["train_texts"]
TRAIN_ANN_RAW = config["paths"]["ner"]["raw"]["train"]["train_ann.tsv"]
DEV_TEXT_RAW = config["paths"]["ner"]["raw"]["dev"]["dev_texts"]
DEV_ANN_RAW = config["paths"]["ner"]["raw"]["dev"]["dev_ann.tsv"]

# Load text+ann data
print("\n Loading data...")
train_data = load_txt_ann(TRAIN_TEXT_RAW, TRAIN_ANN_RAW)
dev_data = load_txt_ann(DEV_TEXT_RAW, DEV_ANN_RAW)
print("✅ Data loaded.")

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
