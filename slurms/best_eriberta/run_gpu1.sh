#!/usr/bin/env bash

export TRANSFORMERS_CACHE="/ncache/hub"
export HF_HUB_CACHE="/ncache/hub"
export HF_HOME="/ncache/hub"
export HF_DATASETS_CACHE="/ncache/hub/datasets"
export HF_MODULES_CACHE="/ncache/hub/modules"
export PATH="/ikerlariak/idelaiglesia004/conda-envs/llms/bin:$PATH"

BASE_PATH="/ikerlariak/idelaiglesia004/eriberta"
cd "$BASE_PATH" || exit

CONFIG_PATH="config/ner/config_ner_meddoprof_subtrack1-ner.yaml"
python run_ner_pipeline.py --config_path "$CONFIG_PATH" --force_fine_tuning --force_tokenize --model="eriberta" --seed=142
python delete_checkpoints.py --config_path "$CONFIG_PATH" --model="eriberta"

CONFIG_PATH="config/ner/config_ner_meddoprof_subtrack2-class.yaml"
python run_ner_pipeline.py --config_path "$CONFIG_PATH" --force_fine_tuning --force_tokenize --model="eriberta" --seed=142
python delete_checkpoints.py --config_path "$CONFIG_PATH" --model="eriberta"

CONFIG_PATH="config/ner/config_ner_pharmaconer_fullcases.yaml"
python run_ner_pipeline.py --config_path "$CONFIG_PATH" --force_fine_tuning --force_tokenize --model="eriberta" --seed=42
python delete_checkpoints.py --config_path "$CONFIG_PATH" --model="eriberta"
