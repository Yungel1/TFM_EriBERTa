import os
import pandas as pd

from datasets import Dataset

import os
from datasets import Dataset


def create_hf_dataset_from_brats(text_ann_folder):
    data = []

    # Iterate over all files in the folder
    for file in os.listdir(text_ann_folder):
        if file.endswith(".txt"):
            # Extract the article_id from the filename (without extension)
            article_id = os.path.splitext(file)[0]
            text_path = os.path.join(text_ann_folder, file)

            # Read the content of the text file
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            entities = []
            # Find the corresponding .ann file (same name as the .txt)
            ann_file = os.path.join(text_ann_folder, article_id + ".ann")
            if os.path.exists(ann_file):
                with open(ann_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Process only annotation lines that start with "T"
                        if line.startswith("T"):
                            # Expected format (Brat style):
                            # T1  <tab>  Label Start End[; Start End...]  <tab>  Text
                            parts = line.split("\t")
                            if len(parts) < 2:
                                continue
                            # The second part contains the annotation info
                            annotation_info = parts[1].split()
                            label = annotation_info[0]
                            # Start and end indices might appear in pairs for discontinuous annotations
                            try:
                                token_list = annotation_info[1:]
                                for i in range(0, len(token_list), 2):
                                    start = int(token_list[i])
                                    end = int(token_list[i + 1])
                                    entities.append({
                                        "label": label,
                                        "start": start,
                                        "end": end
                                    })
                            except Exception as e:
                                print(f"Error parsing annotation in {ann_file}: {e}")

            data.append({"text": text, "entities": entities})

    # Convert the list of data to a Hugging Face Dataset
    dataset = Dataset.from_list(data)
    return dataset

def tokenize_and_align_labels(example, tokenizer, label_map, max_length=512, stride=128):
    tokenized_inputs = tokenizer(
        example["text"],
        truncation=False,
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False
    )

    offset_mappings = tokenized_inputs.pop("offset_mapping")
    labels = []

    for i, offsets in enumerate(offset_mappings):
        input_ids = tokenized_inputs["input_ids"][i]
        label_ids = [-100] * len(input_ids)

        entity_spans = []
        for entity in example["entities"]:
            entity_spans.append((entity["start"], entity["end"], entity["label"]))

        for idx, (token_start, token_end) in enumerate(offsets):
            if token_start == 0 and token_end == 0:
                continue

            for start, end, entity_label in entity_spans:
                if token_start >= end:
                    continue

                if start <= token_start < end or start < token_end <= end:
                    if label_ids[idx] == -100:
                        label_ids[idx] = label_map[f'B-{entity_label}']
                    else:
                        label_ids[idx] = label_map[f'I-{entity_label}']
                    break

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def process_dataset(dataset):
    if not isinstance(dataset, Dataset):
        dataset = Dataset.from_list(dataset)

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=False,
        remove_columns=["text", "entities"],
        desc="Tokenizing and aligning labels"
    )

    tokenized_dataset = tokenized_dataset.filter(lambda x: x["input_ids"] is not None)
    return tokenized_dataset

