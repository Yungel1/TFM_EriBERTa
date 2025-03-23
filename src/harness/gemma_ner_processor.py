from datasets import Dataset
import json


class GemmaNER:
    def __init__(
        self,
        no_ent_string: str = "&&NOENT&&",
        ner_entity_separator: str = ",",
        ner_type_separator: str = "$",
        ner_entity_list: list = None,
        gen_prefix: str = "Entidades: "
    ):
        """
        Initializes the GemmaNER class with the required parameters for processing.
        """
        if ner_entity_list is None:
            ner_entity_list = []
        self.no_ent_string = no_ent_string
        self.ner_entity_separator = ner_entity_separator
        self.ner_type_separator = ner_type_separator
        self.ner_entity_list = ner_entity_list
        self.ner_mapping = {t: i for i, t in enumerate(self.ner_entity_list)}
        self.gen_prefix = gen_prefix

    def _process_example(self, example: dict) -> dict:
        """
        Processes an individual example for entity extraction.

        Args:
            example: A dictionary with keys "text", "entities", and "article_id".

        Returns:
            A dictionary with keys "id", "text", "target", and processed "entities".
        """
        text = example["text"]
        sorted_entities = sorted(
            example["entities"],
            key=lambda x: (x["start"], x["end"])
        )
        new_entities = []
        target_entries = []

        for ent in sorted_entities:
            # Extract entity text from the original text
            entity_text = text[ent["start"]:ent["end"]]
            # Append the processed entity
            new_entities.append({
                "entity_text": entity_text.strip(),
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"]
            })
            # Append the target entry for the entity
            target_entries.append(f"{entity_text.strip()}{self.ner_type_separator}{ent['label']}")

        return {
            "id": example["article_id"],
            "text": text,
            "target": self.ner_entity_separator.join(target_entries) if target_entries else self.no_ent_string,
            "entities": new_entities,
        }

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Processes a dataset by applying the process_example function to each entry.

        Args:
            dataset: A Dataset object from the 'datasets' library.

        Returns:
            The processed Dataset.
        """
        return dataset.map(
            self._process_example, remove_columns=dataset.column_names
        )

    def _ner_process_raw_output(self, llm_result: str) -> list:
        """
        Parses raw model output into a list of (entity_text, entity_type) tuples.

        Args:
            llm_result: Raw model output string (e.g., "text$TYPE,text2$TYPE2").

        Returns:
            A list of (entity_text, entity_type) tuples.
        """
        if self.no_ent_string in llm_result:
            return []
        if llm_result.strip() == "":
            return [("WRONG", "O")]  # Special error case

        entities = []
        for entity_str in llm_result.split(self.ner_entity_separator):
            parts = [s.strip() for s in entity_str.split(self.ner_type_separator) if s]
            # Handle malformed entries
            entity_text = parts[0].lower() if len(parts) >= 1 else ""
            entity_type = parts[1].upper() if len(parts) >= 2 else ""

            if entity_text and entity_type in self.ner_entity_list:
                entities.append((entity_text, entity_type))
        return entities

    def _ner_align_entities(self, pred_entities: list, gold_entities: list) -> tuple:
        """
        Aligns predicted and gold entities for evaluation.

        Matching priority:
          1. Perfect matches (text and type)
          2. Text matches with type mismatch
          3. Remaining entities are mapped to the 'O' class

        Returns:
            A tuple of (aligned_pred_labels, aligned_gold_labels).
        """
        pred_copy = pred_entities.copy()
        gold_copy = gold_entities.copy()
        aligned_pred = []
        aligned_gold = []

        # Stage 1: Perfect matches
        for pred in pred_entities:
            for i, gold in enumerate(gold_copy):
                if pred == gold:
                    aligned_pred.append(self.ner_mapping[pred[1]])
                    aligned_gold.append(self.ner_mapping[gold[1]])
                    gold_copy.pop(i)
                    pred_copy.remove(pred)
                    break

        # Stage 2: Text matches with type mismatch
        remaining_pred = pred_copy.copy()
        for pred in remaining_pred:
            for i, gold in enumerate(gold_copy):
                if pred[0] == gold[0]:
                    aligned_pred.append(self.ner_mapping[pred[1]])
                    aligned_gold.append(self.ner_mapping[gold[1]])
                    gold_copy.pop(i)
                    pred_copy.remove(pred)
                    break

        # Stage 3: Map remaining entities to 'O' class
        for pred in pred_copy:
            aligned_pred.append(self.ner_mapping[pred[1]])
            aligned_gold.append(self.ner_mapping['O'])

        for gold in gold_copy:
            aligned_pred.append(self.ner_mapping['O'])
            aligned_gold.append(self.ner_mapping[gold[1]])

        return aligned_pred, aligned_gold

    def ner_process_results(self, doc: dict, results) -> dict:
        """
        Processes the results of the Named Entity Recognition task.

        Args:
            doc: A dictionary containing at least the "target" key with the true entities string.
            results: The model output (string or list) containing the predicted entities.

        Returns:
            A dictionary with the key "f1" containing a tuple of (predicted_labels, gold_labels).
        """
        # Extract model output string (handle list inputs)
        if isinstance(results, list):
            results = results[0]  # Assume the first element contains the prediction string

        # Remove "Entidades: " prefix if present
        if results.startswith(self.gen_prefix):
            results = results[len(self.gen_prefix):]

        # Process both gold (ground truth) and predicted entities
        gold = self._ner_process_raw_output(doc["target"])
        pred = self._ner_process_raw_output(results)

        # Special case: both empty predictions
        if not pred and not gold:
            return {"f1": ([self.ner_mapping['O']], [self.ner_mapping['O']])}

        # Align entity lists for metric calculation
        if len(gold) <= len(pred):
            gold_labels, pred_labels = self._ner_align_entities(gold, pred)
        else:
            pred_labels, gold_labels = self._ner_align_entities(pred, gold)

        # Safety check - alignment must match list lengths
        assert len(gold_labels) == len(pred_labels)
        return {"f1": (pred_labels, gold_labels)}


