#!/usr/bin/env bash
#SBATCH --job-name=eriberta_evaluation
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/idelaiglesia004/eriberta/.slurm/%x.%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=iker.delaiglesia@ehu.eus
#SBATCH --time=2-0

export TRANSFORMERS_CACHE="/ncache/hub"
export HF_HUB_CACHE="/ncache/hub"
export HF_HOME="/ncache/hub"
export HF_DATASETS_CACHE="/ncache/hub/datasets"
export HF_MODULES_CACHE="/ncache/hub/modules"
export HF_DATASETS_DOWNLOADED_DATASETS_PATH="/ncache/hub/datasets/downloads"
export HF_DATASETS_EXTRACTED_DATASETS_PATH="/ncache/hub/datasets/downloads/extracted"

export PATH="/ikerlariak/idelaiglesia004/conda-envs/llms/bin":"$PATH"

BASE_PATH="/ikerlariak/idelaiglesia004/eriberta"
cd $BASE_PATH || exit

CONFIG_PATH="config/ner/config_ner_meddoprof_subtrack2-class.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="mdeberta" --runs=5

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="mdeberta"

CONFIG_PATH="config/ner/config_ner_multicardioner_track1.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="mdeberta" --runs=5

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="mdeberta"

CONFIG_PATH="config/ner/config_ner_pharmaconer_sentences.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="mdeberta" --runs=5

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="mdeberta"

CONFIG_PATH="config/ner/config_ner_symptemist.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="mdeberta" --runs=5

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="mdeberta"
