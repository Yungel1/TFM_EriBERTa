from datasets import Dataset


NO_ENT_STRING = "&&NOENT&&"
NER_ENTITY_SEPARATOR = ","
NER_TYPE_SEPARATOR = "$"
NER_ENTITY_LIST = ['O', 'PROCEDIMIENTO']
NER_MAPPING = {t: i for i, t in enumerate(NER_ENTITY_LIST)}
GEN_PREFIX = 'Entidades: '


# PROCESS INPUT FUNCTIONS

def _process_example(example: dict) -> dict:
    """
    Processes an individual example for entity extraction.

    Args:
        example: A dictionary with keys "text", "entities", and "id".

    Returns:
        A dictionary with keys "id", "text" and "target".
    """
    text = example["text"]
    sorted_entities = sorted(example["entities"], key=lambda x: (x["start"], x["end"]))

    target_entries = [
        f"{text[int(ent['start']): int(ent['end'])].strip()}{NER_TYPE_SEPARATOR}{ent['ent_type']}"
        for ent in sorted_entities
    ]

    return {
        "id": example["id"],
        "text": text,
        "target": NER_ENTITY_SEPARATOR.join(target_entries) if target_entries else NO_ENT_STRING,
    }


def process_dataset(dataset: Dataset) -> Dataset:
    """
    Processes a dataset by applying the process_example function to each entry.

    Args:
        dataset: A Dataset object from the 'datasets' library.

    Returns:
        The processed Dataset.
    """
    return dataset.map(
        _process_example, remove_columns=dataset.column_names
    )


# EVALUATION FUNCTIONS


def _ner_process_raw_output(llm_result: str) -> list:
    """
    Parses raw model output into a list of (entity_text, entity_type) tuples.

    Args:
        llm_result: Raw model output string (e.g., "text$TYPE,text2$TYPE2").

    Returns:
        A list of (entity_text, entity_type) tuples.
    """
    if NO_ENT_STRING in llm_result:
        return []
    if llm_result.strip() == "":
        return [("WRONG", "O")]  # Special error case

    entities = []
    for entity_str in llm_result.split(NER_ENTITY_SEPARATOR):
        parts = [s.strip() for s in entity_str.split(NER_TYPE_SEPARATOR) if s]
        # Handle malformed entries
        entity_text = parts[0].lower() if len(parts) >= 1 else ""
        entity_type = parts[1].upper() if len(parts) >= 2 else ""

        if entity_text and entity_type in NER_ENTITY_LIST:
            entities.append((entity_text, entity_type))
    return entities


def _ner_align_entities(pred_entities: list, gold_entities: list) -> tuple:
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
                aligned_pred.append(NER_MAPPING[pred[1]])
                aligned_gold.append(NER_MAPPING[gold[1]])
                gold_copy.pop(i)
                pred_copy.remove(pred)
                break

    # Stage 2: Text matches with type mismatch
    remaining_pred = pred_copy.copy()
    for pred in remaining_pred:
        for i, gold in enumerate(gold_copy):
            if pred[0] == gold[0]:
                aligned_pred.append(NER_MAPPING[pred[1]])
                aligned_gold.append(NER_MAPPING[gold[1]])
                gold_copy.pop(i)
                pred_copy.remove(pred)
                break

    # Stage 3: Map remaining entities to 'O' class
    for pred in pred_copy:
        aligned_pred.append(NER_MAPPING[pred[1]])
        aligned_gold.append(NER_MAPPING['O'])

    for gold in gold_copy:
        aligned_pred.append(NER_MAPPING['O'])
        aligned_gold.append(NER_MAPPING[gold[1]])

    return aligned_pred, aligned_gold


def ner_process_results(doc: dict, results) -> dict:
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
    if results.startswith(GEN_PREFIX):
        results = results[len(GEN_PREFIX):]

    # Process both gold (ground truth) and predicted entities
    gold = _ner_process_raw_output(doc["target"])
    pred = _ner_process_raw_output(results)

    # Special case: both empty predictions
    if not pred and not gold:
        return {"f1": ([NER_MAPPING['O']], [NER_MAPPING['O']])}

    # Align entity lists for metric calculation
    if len(gold) <= len(pred):
        gold_labels, pred_labels = _ner_align_entities(gold, pred)
    else:
        pred_labels, gold_labels = _ner_align_entities(pred, gold)

    # Safety check - alignment must match list lengths
    assert len(gold_labels) == len(pred_labels)
    return {"f1": (pred_labels, gold_labels)}