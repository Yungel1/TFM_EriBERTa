#!/usr/bin/env bash

export TRANSFORMERS_CACHE="/ncache/hub"
export HF_HUB_CACHE="/ncache/hub"
export HF_HOME="/ncache/hub"
export HF_DATASETS_CACHE="/ncache/hub/datasets"
export HF_MODULES_CACHE="/ncache/hub/modules"
export PATH="/ikerlariak/idelaiglesia004/conda-envs/llms/bin:$PATH"

BASE_PATH="/ikerlariak/idelaiglesia004/eriberta"
cd "$BASE_PATH" || exit

# Datasets y tipo de modelo (public o private)
declare -A DATASETS_MODELS=(
#  ["CANTEMIST"]="public"
#  ["DisTEMIST"]="public"
#  ["MEDDOCAN"]="public"
#  ["MEDDOPROF/subtrack1-ner"]="public"
#  ["MEDDOPROF/subtrack2-class"]="public"
#  ["PharmacoNER/Full_Cases"]="public"
  ["CodiEsp"]="public"
  ["MedProcNER"]="public"
  ["MultiCardioNER/track1"]="public"
  ["MultiCardioNER/track2"]="public"
  ["PharmacoNER/Sentences"]="public"
  ["SympTEMIST"]="public"
  ["berdeak"]="public"
  ["ehr_sekzioak/con_cabeceras"]="public"
  ["ehr_sekzioak/sin_cabeceras"]="public"
)

for dataset in "${!DATASETS_MODELS[@]}"; do
  model_type="${DATASETS_MODELS[$dataset]}"
  model_path="best_models/${dataset}/${model_type}/best_model"
  output_dir="results/inference/EriBERTa/${dataset}"
  python run_ner_inference.py \
    --model_path "$model_path" \
    --text_file_dir datasets/inference \
    --output_dir "$output_dir"
done
