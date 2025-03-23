
NO_ENT_STRING = "&&NOENT&&"
NER_ENTITY_SEPARATOR = ","
NER_TYPE_SEPARATOR = "$"
NER_ENTITY_LIST = ['O', 'NO_NORMALIZABLES', 'NORMALIZABLES', 'PROTEINAS', 'UNCLEAR']
NER_MAPPING = {t: i for i, t in enumerate(NER_ENTITY_LIST)}

GEN_PREFIX = 'Entidades: '


def _ner_process_raw_output(llm_result: str) -> list[tuple]:
    """Parse raw model output into structured entity tuples.

    Args:
        llm_result: Raw string from model (e.g., "text$TYPE,text2$TYPE2")

    Returns:
        List of (entity_text, entity_type) tuples
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


def ner_align_entities(pred_entities: list, gold_entities: list) -> tuple[list, list]:
    """Align predicted and gold entities for evaluation.

    Matching priority:
    1. Perfect matches (text + type)
    2. Text matches with type mismatch
    3. Remaining entities mapped to 'O' class

    Returns:
        Tuple of (aligned_pred_labels, aligned_gold_labels)
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


def ner_process_results(doc, results):
    """
    Process the results of the Named Entity Recognition task.
    """
    # Extract model output string (handle list inputs)
    if isinstance(results, list):
        results = results[0]  # Assume first element contains the prediction string

    # Remove "Entities: " prefix if present
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
        gold_labels, pred_labels = ner_align_entities(gold, pred)
    else:
        pred_labels, gold_labels = ner_align_entities(pred, gold)

    # Safety check - alignment must match list lengths
    assert len(gold_labels) == len(pred_labels)
    return {"f1": (pred_labels, gold_labels)}
