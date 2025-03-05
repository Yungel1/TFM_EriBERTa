import os
import pandas as pd

from datasets import Dataset


class NERPreprocessor:
    def __init__(self, tokenizer, label2id, id2label, max_length=512, stride=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.label2id = label2id
        self.id2label = id2label

    def tokenize_and_align_labels(self, example):
        # Tokenize without truncation to handle overflows and obtain the offsets
        tokenized_inputs = self.tokenizer(
            example["text"],
            truncation=False,
            return_overflowing_tokens=True,
            stride=self.stride,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=False
        )

        # TODO Hay que mirar si hay que ordenar, o no
        # Retrieve the entities (assumed to be sorted by their starting position)
        sorted_entities = sorted(example["entities"], key=lambda ent: (ent["start"], -ent["end"]))

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
        return tokenized_inputs

    def tokenize_dataset(self, dataset):
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


def flatten_example(examples, indices):
    dataset_flatenned = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'offset_mapping': [],
        'overflow_to_sample_mapping': []
    }

    for i, idx in enumerate(indices):
        for j in range(len(examples['input_ids'][i])):
            dataset_flatenned['input_ids'].append(examples['input_ids'][i][j])
            dataset_flatenned['attention_mask'].append(examples['attention_mask'][i][j])
            dataset_flatenned['labels'].append(examples['labels'][i][j])
            dataset_flatenned['offset_mapping'].append(examples['offset_mapping'][i][j])
            dataset_flatenned['overflow_to_sample_mapping'].append(idx)

    return dataset_flatenned


def flatten_dataset(nested_dataset):

    # with_indices needed to identify original example
    flattened_dataset = nested_dataset.map(flatten_example, with_indices=True, batched=True)

    return flattened_dataset


# TODO Borrar, es una versión lenta
def flatten_dataset1(nested_dataset):
    flat_data = []
    for i in range(len(nested_dataset['input_ids'])):
        for j in range(len(nested_dataset['input_ids'][i])):
            flat_data.append({
                'input_ids': nested_dataset['input_ids'][i][j],
                'attention_mask': nested_dataset['attention_mask'][i][j],
                'labels': nested_dataset['labels'][i][j],
                'offset_mapping': nested_dataset['offset_mapping'][i][j],
                'overflow_to_sample_mapping': i  # Original example index
            })
    dataset = Dataset.from_list(flat_data)
    return dataset



