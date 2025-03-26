import argparse
import logging
import os
import re

from src.utils.config_loader import load_config

# Logs configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def __get_args():
    # Manage arguments
    parser = argparse.ArgumentParser(description="Tokenization and data preparation for fine-tuning EriBERTa")
    parser.add_argument("--model", type=str, default="eriberta", choices=["eriberta"], help="Model to use")
    parser.add_argument("--config_path", type=str, default="config/config_casimedicos_ner.yaml", help="Config path")
    parser.add_argument("--folder_type", type=str, default="run", choices=["run", "wandb"], help="Checkpoints folder type")
    return parser.parse_args()


def delete_huggingface_checkpoints():
    # Arguments
    args = __get_args()

    # Selected arg model
    arg_model = args.model

    # Load NER general configs
    config = load_config(args.config_path)

    # RESULTS_PATH from configuration file
    RESULTS_PATH = config["models"][arg_model]["results"]

    # Get all run directories (e.g., run_0, run_1, etc.) or wandb folder
    folder_type = args.folder_type
    if folder_type == "run":
        run_folders = [os.path.join(RESULTS_PATH, f) for f in os.listdir(RESULTS_PATH) if re.match(r'run_\d+', f)]
    else:
        run_folders = [os.path.join(RESULTS_PATH, f) for f in os.listdir(RESULTS_PATH) if re.match('wandb', f)]
    checkpoint_folders = []

    # Search for checkpoint folders inside each run directory
    for run_folder in run_folders:
        if os.path.isdir(run_folder):
            checkpoint_folders.extend(
                [os.path.join(run_folder, f) for f in os.listdir(run_folder) if re.match(r'checkpoint-\d+', f)])

    # If no checkpoint folders are found, exit
    if not checkpoint_folders:
        logger.info("No checkpoints found to delete.")
        return

    for folder in checkpoint_folders:
        # Delete all files and subdirectories inside the checkpoint folder
        for root, dirs, files in os.walk(folder, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(folder)  # Finally, remove the checkpoint folder itself
    logger.info(f"{folder_type} checkpoints from {RESULTS_PATH} successfully deleted.")


if __name__ == "__main__":
    delete_huggingface_checkpoints()
