


def extract_label_map(dataset):
    unique_labels = set()

    # Extract labels from dataset
    for example in dataset:
        for entity in example["entities"]:
            unique_labels.add(entity["label"])

    # Sort for consistent ordering
    unique_labels = sorted(unique_labels)

    label_map = {"O": 0}
    for i, entity in enumerate(unique_labels, start=1):
        label_map[f"B-{entity}"] = i * 2 - 1
        label_map[f"I-{entity}"] = i * 2

    return label_map