
# TFM_EriBERTa

**Autor:** Adrián Sánchez Freire

Este repositorio contiene el código correspondiente al TFM **"Benchmarking de Modelos de Lenguaje Discriminativos en el Ámbito Clínico y Biomédico"**, cuyo objetivo es realizar un *benchmarking* de modelos de reconocimiento de entidades médicas (MER) en español. Se actualizan los resultados de EriBERTa y se comparan tanto con modelos discriminativos *state-of-the-art* como con un modelo de lenguaje de gran tamaño (LLM). Esto permite analizar el rendimiento de EriBERTa frente a modelos de su misma categoría y frente a arquitecturas generativas modernas.

---

## Estructura del Repositorio

- **`config/`**: Contiene los archivos YAML que definen las rutas de datos y resultados, modelos a evaluar, hiperparámetros y nombres de proyectos para seguimiento en [Weights & Biases (wandb)](https://wandb.ai/site/).
- **`harness/`**: Contiene todos los archivos necesarios para evaluar el LLM sobre cada dataset con [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).
- **`results/`**: Resultados generados por los experimentos, solo las métricas. Los modelos y predicciones son muy pesados como para guardarlos en el repositorio.  
- **`src/`**: Código fuente del pipeline de preprocesamiento, *fine-tuning* y evaluación. También se encuentra aquí todo el código fuente para realizar la optimización mediante wandb.
- **`run_ner_pipeline.py`**: Script principal para la ejecución del preproceso, entrenamiento (*fine-tuning*) y evaluación. También se usa este para la optimización de hiperparámetros. 
- **`delete_checkpoints.py`**: Script auxiliar para eliminar checkpoints y ahorrar espacio en disco.  
- **`requirements.txt`**: Lista de dependencias necesarias para que el proyecto funcione.

---

## Argumentos del Script `run_ner_pipeline.py`

| Argumento | Tipo | Obligatorio | Valor por defecto | Descripción                                                                                                |
|----------|------|-------------|-------------------|------------------------------------------------------------------------------------------------------------|
| `--model` | `str` | No | `"eriberta"` | Modelo a utilizar. Opciones: `eriberta`, `eriberta_private`, `longformer`, `bsc`, `clin_x_es`, `mdeberta`. |
| `--force_tokenize` | `flag` | No | `False` | Fuerza la tokenización, incluso si ya existe una versión tokenizada.                                       |
| `--force_fine_tuning` | `flag` | No | `False` | Fuerza el *fine-tuning*, aunque ya exista un modelo entrenado.                                             |
| `--opt_hyperparameters` | `flag` | No | `False` | Ejecuta la optimización de hiperparámetros con Weights & Biases (`wandb`).                                 |
| `--config_path` | `str` | Sí | - | Ruta al archivo `.yaml` de configuración.                                                                  |
| `--runs` | `int` | No | `1` | Número de ejecuciones del fine-tuning. Útil para evaluar estabilidad de resultados.                        |
| `--max_batch_size` | `int` | No | `None` | Tamaño máximo de batch. Si se define, ajusta la acumulación de gradientes automáticamente.                 |
| `--use_global_attention` | `flag` | No | `False` | Activa `global_attention_mask`, útil para modelos como Longformer.                                         |

---

## Argumentos del Script `delete_checkpoints.py`

| Argumento | Tipo | Obligatorio | Valor por defecto | Descripción |
|----------|------|-------------|-------------------|-------------|
| `--model` | `str` | Sí | - | Modelo del cual se eliminarán los checkpoints. |
| `--config_path` | `str` | Sí | - | Ruta al archivo de configuración usado en el experimento. |
| `--folder_type` | `str` | No | `"run"` | Tipo de carpeta donde están los checkpoints: `run` o `wandb`. |

---

## Ejecución del Código

> Sustituye las variables como `$CONFIG_PATH`, `$MODEL`, `$TASK`, etc., por los valores concretos según tu entorno.

### 1. Ejemplo de optimización de hiperparámetros con `wandb`

#### Parámetros:

```bash
MODEL="bsc"
CONFIG_PATH="config/ner/config_ner_cantemist.yaml"
```

#### Ejecución:

```bash
python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--opt_hyperparameters --force_fine_tuning --force_tokenize --model="$MODEL"
```

#### Limpieza de checkpoints:

```bash
python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="$MODEL" --folder_type="wandb"
```

---

### 2. Ejemplo de ejecución completa del pipeline

#### Parámetros:

```bash
MODEL="bsc"
CONFIG_PATH="config/ner/config_ner_cantemist.yaml"
```

#### Ejecución. Incluye tokenización, fine-tuning y evaluación:

```bash
python run_ner_pipeline.py --config_path "$CONFIG_PATH" \
--force_fine_tuning --force_tokenize --model="$MODEL" --runs=5
```

#### Limpieza de checkpoints:

```bash
python delete_checkpoints.py --config_path "$CONFIG_PATH" \
--model="$MODEL"
```

---

### 3. Ejemplo de evaluación del modelo Gemma 3 (LLM)

La evaluación se realiza mediante el framework [**lm-evaluation-harness**](https://github.com/EleutherAI/lm-evaluation-harness) desarrollado por EleutherAI. Esta herramienta permite evaluar modelos de lenguaje en múltiples tareas, incluyendo NER. Consulta su repositorio para más detalles sobre el uso del comando `lm_eval`.

> ⚠️ Nota: En este contexto, la variable `$MODEL` hace referencia al identificador del modelo en [Hugging Face Hub](https://huggingface.co/), no al argumento `--model` del script `run_ner_pipeline.py`.

#### Evaluación:

```bash
MODEL="google/gemma-3-12b-it"
TASK="cantemist-entity_list"
TASKS_PATH="harness"
OUTPUT_PATH="results/gemma/NER/CANTEMIST"

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
```
