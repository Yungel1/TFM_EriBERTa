import os
import pandas as pd

from datasets import Dataset


def extract_label_maps(dataset, unique_labels=None):

    if unique_labels is None:
        unique_labels = set()

        # Extract labels from dataset
        for example in dataset:
            for entity in example["entities"]:
                unique_labels.add(entity["label"])

    # Sort for consistent ordering
    unique_labels = sorted(unique_labels)

    label2id = {"O": 0}
    for i, entity in enumerate(unique_labels, start=1):
        label2id[f"B-{entity}"] = i * 2 - 1
        label2id[f"I-{entity}"] = i * 2

    id2label = {i: l for l, i in label2id.items()}

    return label2id, id2label


class NERPreprocessor:
    def __init__(self, tokenizer, max_length=512, stride=128, label_all_tokens=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.label_all_tokens = label_all_tokens

    def create_hf_dataset_from_brats(self, text_ann_folder):
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

        # Defino los mapeos de los labels
        self.label2id, self.id2label = extract_label_maps(dataset)

        return dataset

    def tokenize_and_align_labels(self, example):
        tokenized_inputs = self.tokenizer(
            example["text"],
            truncation=False,
            return_overflowing_tokens=True,
            stride=self.stride,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=False
        )

        sorted_entities = sorted(example["entities"], key=lambda ent: ent["start"])
        num_entities = len(sorted_entities)

        labels = []
        for offsets in tokenized_inputs["offset_mapping"]:
            label_seq = [-100] * len(offsets)
            ent_idx = 0

            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start == 0 and token_end == 0:
                    continue

                label_seq[idx] = self.label2id['O']

                while ent_idx < num_entities and sorted_entities[ent_idx]["end"] <= token_start:
                    ent_idx += 1

                if ent_idx < num_entities:
                    entity = sorted_entities[ent_idx]
                    if token_end > entity["start"] and token_start < entity["end"]:
                        label_seq[idx] = (
                            self.label2id[f'B-{entity["label"]}']
                            if token_start == entity["start"]
                            else (self.label2id[f'I-{entity["label"]}'] if self.label_all_tokens else -100)
                        )

            labels.append(label_seq)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def process_dataset(self, dataset):
        if not isinstance(dataset, Dataset):
            dataset = Dataset.from_list(dataset)

        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing and aligning labels"
        )

        tokenized_dataset = tokenized_dataset.filter(lambda x: x["input_ids"] is not None)
        return tokenized_dataset

