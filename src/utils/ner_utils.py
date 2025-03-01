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

    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label
