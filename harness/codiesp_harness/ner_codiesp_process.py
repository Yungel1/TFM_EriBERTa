from datasets import Dataset


NO_ENT_STRING = "&&NOENT&&"
NER_ENTITY_SEPARATOR = ","
NER_TYPE_SEPARATOR = "$"
NER_ENTITY_LIST = ['O', 'DIAGNOSTICO', 'PROCEDIMIENTO']
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
        _process_example#, remove_columns=dataset.column_names
    )


# EVALUATION FUNCTIONS


def _ner_gold_to_target(x: list) -> list:
    """
    Convert the gold entities to the target format according to the NER_MAPPING
    """
    res = [NER_MAPPING[e["ent_type"]] for e in x]
    return res


def _ner_process_raw_output(llm_result: str) -> list[tuple]:
    if NO_ENT_STRING in llm_result:
        return []
    if llm_result == "":
        return ["WRONG"]
    tmp_results = llm_result.split(NER_ENTITY_SEPARATOR)
    results = []
    for res in tmp_results:
        r = res.strip()
        # split on type separator
        r_text = ""
        r_type = ""
        r_splitted = r.split(NER_TYPE_SEPARATOR)
        if len(r_splitted) < 2:
            r_text = r_splitted[0]
            r_type = ""
        else:
            r_text = r_splitted[0]
            r_type = r_splitted[1]
        if r_text != "":
            results.append((r_text, r_type.upper()))
    return results


def ner_process_results(doc, results):
    """
    Process the results of the Named Entity Recognition task
    """
    # each document has a list of entities with the following format:
    # [{"entity_text": "string", "type": "string"}]
    gold = doc["entities"]
    raw_results = results[0]
    results = _ner_process_raw_output(raw_results)

    gold_labels = _ner_gold_to_target(gold)
    res_labels = [0] * len(gold_labels)
    matched_gold_idx = []

    if len(results) > len(gold):
        for r in results:
            r_text = r[0]
            r_type = r[1]
            for i in range(len(gold)):
                if r_text == gold[i]["text"] and r_type == gold[i]["ent_type"]:
                    res_labels[i] = NER_MAPPING[r_type]
                    matched_gold_idx.append(i)
        # Since we have more results than gold, we artificially set to false positive the remaining labels
        # extend gold label list
        for i in range(len(results) - len(gold)):
            gold_labels.append(3)
            res_labels.append(2)
    elif len(results) == 0 and len(gold) == 0:
        res_labels = [3]
        gold_labels = res_labels
    else:  # len(results) <= len(gold)
        for r in results:
            r_text = r[0]
            r_type = r[1]
            for i in range(len(gold)):
                if r_text == gold[i]["text"] and r_type == gold[i]["ent_type"]:
                    res_labels[i] = NER_MAPPING[r_type]
                    matched_gold_idx.append(i)
        # we map all wrong predictions to the "O" class
        for i in range(len(gold_labels)):
            if i in matched_gold_idx:
                continue
            if gold_labels[i] == 1:
                res_labels[i] = 3
            elif gold_labels[i] == 0:
                res_labels[i] = 3
            else:
                res_labels[i] = 3

    assert len(gold_labels) == len(res_labels)
    return {"f1": (res_labels, gold_labels)}
