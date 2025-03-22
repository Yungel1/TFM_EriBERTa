import json
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

            data.append({"article_id": article_id, "text": text, "entities": entities})

    # Convert the list of data to a Hugging Face Dataset
    dataset = Dataset.from_list(data)

    return dataset


def create_hf_dataset_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for entry in data:
        article_id = entry.get("id", "unknown")
        text = entry.get("text", "")
        entities = []

        for ent in entry.get("entities", []):
            try:
                entities.append({
                    "label": ent["ent_type"],
                    "start": int(ent["start"]),
                    "end": int(ent["end"])
                })
            except Exception as e:
                print(f"Error parsing entity in {article_id}: {e}")

        formatted_data.append({"article_id": article_id, "text": text, "entities": entities})

    dataset = Dataset.from_list(formatted_data)
    return dataset
