import time

from src.data_preprocessing.ner_preprocess import create_hf_dataset_from_brats
from src.utils.config_loader import load_ner_config

# Load NER general configs
config = load_ner_config("config.yaml")

# Paths from configuration file
TRAIN_RAW = config["paths"]["ner"]["raw"]["train"]
DEV_RAW = config["paths"]["ner"]["raw"]["dev"]
TRAIN_PROCESSED = config["paths"]["ner"]["processed"]["train"]
DEV_PROCESSED = config["paths"]["ner"]["processed"]["dev"]

# Load text+ann (BRAT) data into HF Dataset
print("\nLoading data...")
start_time = time.time()
train_dataset = create_hf_dataset_from_brats(TRAIN_RAW)
dev_dataset = create_hf_dataset_from_brats(DEV_RAW)
print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds.\n")
# TODO Prueba (borrar cuando se compruebe)
print(train_dataset[4])

# Save HF Dataset to disk
train_dataset.save_to_disk(TRAIN_PROCESSED)
dev_dataset.save_to_disk(DEV_PROCESSED)


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
