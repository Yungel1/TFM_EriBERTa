#!/usr/bin/env bash
#SBATCH --job-name=gemma_evaluation_harness
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/idelaiglesia004/eriberta/.slurm/%x.%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=iker.delaiglesia@ehu.eus
#SBATCH --time=1-0

export TRANSFORMERS_CACHE="/ncache/hub"
export HF_HUB_CACHE="/ncache/hub"

export PATH="/ikerlariak/idelaiglesia004/conda-envs/llms/bin":"$PATH"

BASE_PATH="/ikerlariak/idelaiglesia004/eriberta"
cd $BASE_PATH || exit

MODEL="google/gemma-3-12b-it"
TASK="meddoprof_subtrack1-ner-entity_list"
TASKS_PATH="harness"
OUTPUT_PATH="results/gemma/NER/MERDDOPROF/subtrack1-ner"

# Ejecutar lm-evaluation-harness con el modelo y tasks definidos. #--num_fewshot $fs \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL \
  --tasks $TASK \
  --include_path $TASKS_PATH \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --device cuda \
  --batch_size 1 \
  --apply_chat_template \
  --log_samples \
  --output_path $OUTPUT_PATH

TASK="meddoprof_subtrack2-class-entity_list"
OUTPUT_PATH="results/gemma/NER/MERDDOPROF/subtrack2-class"

# Ejecutar lm-evaluation-harness con el modelo y tasks definidos. #--num_fewshot $fs \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL \
  --tasks $TASK \
  --include_path $TASKS_PATH \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --device cuda \
  --batch_size 1 \
  --apply_chat_template \
  --log_samples \
  --output_path $OUTPUT_PATH

TASK="medprocner-entity_list"
OUTPUT_PATH="results/gemma/NER/MedProcNER"

# Ejecutar lm-evaluation-harness con el modelo y tasks definidos. #--num_fewshot $fs \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL \
  --tasks $TASK \
  --include_path $TASKS_PATH \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --device cuda \
  --batch_size 1 \
  --apply_chat_template \
  --log_samples \
  --output_path $OUTPUT_PATH

TASK="multicardioner_track1-entity_list"
OUTPUT_PATH="results/gemma/NER/MultiCardioNER/track1"

# Ejecutar lm-evaluation-harness con el modelo y tasks definidos. #--num_fewshot $fs \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL \
  --tasks $TASK \
  --include_path $TASKS_PATH \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --device cuda \
  --batch_size 1 \
  --apply_chat_template \
  --log_samples \
  --output_path $OUTPUT_PATH