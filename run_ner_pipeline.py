import os
import time
from src.data_preprocessing import ner_preprocess
from src.utils import data_loader
from src.utils.config_loader import load_ner_config

# Configuración general
config = load_ner_config("config.yaml")

# Rutas desde el archivo de configuración
TRAIN_TEXT_RAW = "data/ner/raw/train/train_texts/"
TRAIN_ANN_RAW = "data/ner/raw/train/train_ann.tsv"
DEV_TEXT_RAW = "data/ner/raw/dev/dev_texts/"
DEV_ANN_RAW = "data/ner/raw/dev/dev_ann.tsv"

print(DEV_ANN_RAW)

# # Preprocesamiento de los datos
# print("\n Iniciando preprocesamiento de datos...")
# start_time = time.time()
# ner_preprocess(input_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR)
# print(f"✅ Datos preprocesados en {time.time() - start_time:.2f} segundos.")
#
# # Cargar los datos preprocesados
# print("\n Cargando datos...")
# train_data = data_loader(os.path.join(PROCESSED_DATA_DIR, "train.json"))
# dev_data = data_loader(os.path.join(PROCESSED_DATA_DIR, "dev.json"))
# test_data = data_loader(os.path.join(PROCESSED_DATA_DIR, "test.json"))
# print("✅ Datos cargados correctamente.")

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
