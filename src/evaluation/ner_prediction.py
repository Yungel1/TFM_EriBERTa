from collections import defaultdict
import numpy as np

import numpy as np
from collections import defaultdict


def merge_tokens(predictions, labels, offset_mappings, overflow_to_sample_mapping):
    merged_predictions = defaultdict(list)
    merged_labels = defaultdict(list)
    processed_offsets = defaultdict(set)  # Para evitar duplicados debido al stride

    for i, (prediction, label, offsets, original_instance) in enumerate(
            zip(predictions, labels, offset_mappings, overflow_to_sample_mapping)):
        instance_id = original_instance[0]  # Identificar la instancia original

        for j, (p, l, offset) in enumerate(zip(prediction, label, offsets)):
            start, end = offset
            if (start, end) in processed_offsets[instance_id]:
                continue  # Evitar procesar el mismo token más de una vez debido al stride
            processed_offsets[instance_id].add((start, end))

            merged_predictions[instance_id].append(p)
            merged_labels[instance_id].append(l)

    # Convertir a listas ordenadas
    final_predictions = [merged_predictions[i] for i in sorted(merged_predictions.keys())]
    final_labels = [merged_labels[i] for i in sorted(merged_labels.keys())]

    return final_predictions, final_labels


def predict_and_save(trainer, predict_dataset, id2label, compute_metrics, output_dir):
    predictions, labels, _ = trainer.predict(predict_dataset, metric_key_prefix="predict")
    predictions = np.argmax(predictions, axis=2)

    offset_mappings = predict_dataset['offset_mapping']
    overflow_to_sample_mapping = predict_dataset['overflow_to_sample_mapping']

    # Fusionar tokens en palabras, evitando duplicados por el stride
    final_predictions, final_labels = merge_tokens(predictions, labels, offset_mappings, overflow_to_sample_mapping)

    # Convertir índices a etiquetas
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(final_predictions, final_labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(final_predictions, final_labels)
    ]

    # Calcular métricas con las predicciones fusionadas
    merged_metrics = compute_metrics((true_predictions, true_labels), are_predictions_processed=True)

    # Guardar métricas correctamente
    trainer.log_metrics("predict", merged_metrics)
    trainer.save_metrics("predict", merged_metrics)
