#!/usr/bin/env bash
#SBATCH --job-name=eriberta_evaluation
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/idelaiglesia004/eriberta/.slurm/%x.%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=iker.delaiglesia@ehu.eus
#SBATCH --time=2-8

export TRANSFORMERS_CACHE="/ncache/hub"
export HF_HUB_CACHE="/ncache/hub"
export HF_HOME="/ncache/hub"
export HF_DATASETS_CACHE="/ncache/hub/datasets"
export HF_MODULES_CACHE="/ncache/hub/modules"

export PATH="/ikerlariak/idelaiglesia004/conda-envs/llms/bin":"$PATH"

BASE_PATH="/ikerlariak/idelaiglesia004/eriberta"
cd $BASE_PATH || exit

CONFIG_PATH="config/ner/config_ner_cantemist.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="longformer" --max_batch_size=4 --runs=5 --use_global_attention

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="longformer"

CONFIG_PATH="config/ner/config_ner_codiesp.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="longformer" --max_batch_size=4 --runs=5 --use_global_attention

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="longformer"

CONFIG_PATH="config/ner/config_ner_distemist.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="longformer" --max_batch_size=4 --runs=5 --use_global_attention

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="longformer"

CONFIG_PATH="config/ner/config_ner_meddocan.yaml"

python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="longformer" --max_batch_size=4 --runs=5 --use_global_attention

python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="longformer"
