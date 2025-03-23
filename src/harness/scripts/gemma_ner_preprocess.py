from datasets import Dataset
import json


def process_example(example):
    text = example["text"]
    sorted_entities = sorted(
        example["entities"],
        key=lambda x: (x["start"], x["end"])
    )
    new_entities = []
    target_entries = []

    for ent in sorted_entities:
        # Extract entity text
        entity_text = text[ent["start"]:ent["end"]]
        # Append entity
        new_entities.append({
            "entity_text": entity_text.strip(),
            "label": ent["label"],
            "start": ent["start"],
            "end": ent["end"]
        })
        # Append target
        target_entries.append(f"{entity_text.strip()}${ent['label']}")

    return {
        "id": example["article_id"],
        "text": text,
        "target": ",".join(target_entries) if target_entries else "&&NOENT&&",
        "entities": new_entities,
    }


def process_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(
        process_example, remove_columns=dataset.column_names
    )
