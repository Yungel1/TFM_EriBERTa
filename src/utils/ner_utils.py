import json
import os


def extract_label_maps(dataset, label2id_path):

    if os.path.exists(label2id_path):
        # Get label map from path
        label2id = load_label_map(label2id_path)
        print(f"ℹ️ File \"{label2id_path}\" already exists and will be used instead of generating a new one")

    else:
        # Extract labels from dataset
        unique_labels = set()

        for example in dataset:
            for entity in example["entities"]:
                unique_labels.add(entity["label"])

        # Sort for consistent ordering
        unique_labels = sorted(unique_labels)

        # label2id map
        label2id = {"O": 0}
        for i, entity in enumerate(unique_labels, start=1):
            label2id[f"B-{entity}"] = i * 2 - 1
            label2id[f"I-{entity}"] = i * 2

        # Save label2id map
        save_label_map(label2id, label2id_path)

    # id2label map
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


def save_label_map(label_map, label_map_path):
    with open(label_map_path, "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=4)
    print(f"ℹ️ File \"{label_map_path}\" generated.")


def load_label_map(label_map_path):
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"❌ File \"{label_map_path}\" is missing. "
                                f"Run the full process to generate it or add the file manually.")

    with open(label_map_path, "r", encoding="utf-8") as file:
        label_map = json.load(file)
    return label_map


def load_label_maps(label2id_map_path):
    # label2id map
    label2id = load_label_map(label2id_map_path)

    # id2label map
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label
