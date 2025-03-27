import os
from collections import defaultdict

import pandas as pd

from datasets import Dataset


class NERPreprocessor:
    def __init__(self, tokenizer, label2id, id2label, stride=128):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.stride = stride
        self.label2id = label2id
        self.id2label = id2label

    def tokenize_and_align_labels(self, example, indices):
        # Tokenize without truncation to handle overflows and obtain the offsets
        tokenized_inputs = self.tokenizer(
            example["text"][0],
            truncation=True,
            return_overflowing_tokens=True,
            stride=self.stride,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=False
        )

        # TODO Hay que mirar si hay que ordenar, o no
        # Retrieve the entities (assumed to be sorted by their starting position)
        sorted_entities = sorted(example["entities"][0], key=lambda ent: (ent["start"], -ent["end"]))

        labels = []
        for offsets in tokenized_inputs["offset_mapping"]:
            label_seq = [-100] * len(offsets)
            for idx, (token_start, token_end) in enumerate(offsets):
                # Ignore special tokens
                if token_start == 0 and token_end == 0:
                    continue

                # Initially assign the label 'O'
                label_seq[idx] = self.label2id['O']

                # Iterate over all entities
                for entity in sorted_entities:
                    # If the token overlaps with the entity
                    if token_end > entity["start"] and token_start < entity["end"]:
                        # Assign 'B-' if the token starts the entity, 'I-' otherwise
                        label_seq[idx] = (
                            self.label2id[f'B-{entity["label"]}']
                            if token_start == entity["start"]
                            else self.label2id[f'I-{entity["label"]}']
                        )
                        # If there is a nested entity and want to overwrite, comment the break
                        break
            labels.append(label_seq)

        tokenized_inputs["labels"] = labels
        # Map each window to original sample
        tokenized_inputs["overflow_to_sample_mapping"] = [indices] * len(tokenized_inputs["overflow_to_sample_mapping"])
        tokenized_inputs["article_id"] = (
                [example["article_id"][0]] * len(tokenized_inputs["overflow_to_sample_mapping"])
        )
        return tokenized_inputs

    def tokenize_dataset(self, dataset):
        if not isinstance(dataset, Dataset):
            dataset = Dataset.from_list(dataset)

        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            batch_size=1,
            with_indices=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing and aligning labels"
        )

        tokenized_dataset = tokenized_dataset.filter(lambda x: x["input_ids"] is not None)
        return tokenized_dataset
