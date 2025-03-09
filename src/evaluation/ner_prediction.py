import os
from collections import defaultdict
import numpy as np

import numpy as np
from collections import defaultdict


def merge_tokens(predictions, labels, predict_dataset):
    merged_data = defaultdict(lambda: {"predictions": [], "labels": [], "input_ids": [], "offsets": []})
    processed_offsets = defaultdict(set)

    for pred, lab, ids, offsets, original_instance in zip(
            predictions, labels, predict_dataset["input_ids"], predict_dataset["offset_mapping"],
            predict_dataset["overflow_to_sample_mapping"]
    ):
        instance_id = original_instance[0]
        for p, l, token_id, (start, end) in zip(pred, lab, ids, offsets):
            # Only process the token if its offset hasn't been processed before for this instance
            if (start, end) not in processed_offsets[instance_id] and l != -100:
                processed_offsets[instance_id].add((start, end))
                merged_data[instance_id]["predictions"].append(p)
                merged_data[instance_id]["labels"].append(l)
                merged_data[instance_id]["input_ids"].append(token_id)
                merged_data[instance_id]["offsets"].append((start, end))

    # Sort by instance_id (overflow_to_sample_mapping)
    sorted_ids = sorted(merged_data.keys())

    return (
        [merged_data[i]["predictions"] for i in sorted_ids],
        [merged_data[i]["labels"] for i in sorted_ids],
        [merged_data[i]["input_ids"] for i in sorted_ids],
        [merged_data[i]["offsets"] for i in sorted_ids],
    )


def save_conll_predictions(true_input_ids, true_predictions, true_offsets, tokenizer, output_dir):
    output_file = os.path.join(output_dir, "predictions.conll")
    with open(output_file, "w", encoding="utf-8") as f:
        for token_ids, preds, offsets in zip(true_input_ids, true_predictions, true_offsets):
            # Obtain tokens from ids
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            for token, label, offset in zip(tokens, preds, offsets):
                f.write(f"{token} {offset[0]} {label}\n")
            f.write("\n")


def predict_and_save(trainer, predict_dataset, id2label, compute_metrics, tokenizer, output_dir):
    predictions, labels, _ = trainer.predict(predict_dataset, metric_key_prefix="predict")
    predictions = np.argmax(predictions, axis=2)

    # Merge token-level predictions and labels
    final_predictions, final_labels, final_input_ids, final_offsets = merge_tokens(
        predictions,
        labels,
        predict_dataset
    )

    # Assign label, (special tokens not added in merge_tokens, no need to filter them)
    true_predictions = [
        [id2label[p] for p, l in zip(prediction, label)]
        for prediction, label in zip(final_predictions, final_labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(prediction, label)]
        for prediction, label in zip(final_predictions, final_labels)
    ]

    # Compute metrics using the provided function, indicating that the predictions have already been processed.
    merged_metrics = compute_metrics((true_predictions, true_labels), are_predictions_processed=True)

    # Save predictions on CONLL format
    save_conll_predictions(final_input_ids, true_predictions, final_offsets, tokenizer, output_dir)

    # Save metrics
    # trainer.log_metrics("predict", merged_metrics)
    trainer.save_metrics("predict", merged_metrics)
