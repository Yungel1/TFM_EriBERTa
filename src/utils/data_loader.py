import os
import pandas as pd

from datasets import Dataset


def load_txt_ann(text_folder, annotations_file):

    # Load annotations and group by articleID
    annotations = pd.read_csv(
        annotations_file,
        sep="\t",
        header=None,
        names=["articleID", "label", "ICD10-code", "text-reference", "reference-position"]
    )
    grouped_annotations = annotations.groupby("articleID")

    data = []
    # Iterate over each text file
    for text_file in os.listdir(text_folder):
        if text_file.endswith(".txt"):
            article_id = os.path.splitext(text_file)[0]  # extract articleID (such as "1" from "1.txt")
            text_path = os.path.join(text_folder, text_file)

            # Read text file
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            entities = []
            # If annotations exist for that article_id, process them
            if article_id in grouped_annotations.groups:
                article_annotations = grouped_annotations.get_group(article_id)
                for _, row in article_annotations.iterrows():
                    # Handle multiple occurrences splitted by semicolons
                    for ref_pos in row["reference-position"].split(';'):
                        try:
                            start, end = map(int, ref_pos.strip().split())
                            entities.append({
                                "label": row["label"],
                                "start": start,
                                "end": end,
                            })
                        except Exception as e:
                            print(f"Error parsing article positions {article_id} ('{ref_pos}'): {e}")

            data.append({"text": text, "entities": entities})

    # Convert to Hugging face Dataset
    dataset = Dataset.from_list(data)

    return dataset
