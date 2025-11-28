#!/usr/bin/env bash
#SBATCH --job-name=gemma_evaluation_harness
#SBATCH --cpus-per-task=2
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/idelaiglesia004/eriberta/.slurm/%x.%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=iker.delaiglesia@ehu.eus
#SBATCH --time=0-8

export TRANSFORMERS_CACHE="/ncache/hub"
export HF_HUB_CACHE="/ncache/hub"

export PATH="/ikerlariak/idelaiglesia004/conda-envs/llms/bin":"$PATH"

BASE_PATH="/ikerlariak/idelaiglesia004/eriberta"
cd $BASE_PATH || exit

MODEL="meta-llama/Meta-Llama-3.1-8B"
TASK="cantemist-entity_list"
TASKS_PATH="harness"
OUTPUT_PATH="results/gemma/NER/CANTEMIST"

# Ejecutar lm-evaluation-harness con el modelo y tasks definidos. #--num_fewshot $fs \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL \
  --tasks $TASK \
  --include_path $TASKS_PATH \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --device cuda \
  --log_samples \
  --output_path $OUTPUT_PATH
