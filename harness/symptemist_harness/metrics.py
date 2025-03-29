import torch
from sklearn.metrics import f1_score, precision_score, recall_score

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


def _aggreg_ner(predictions):
    """
    Computes micro-averaged F1 for NER, dynamically inferring entity classes.
    Handles false positives on 'O' tokens and missing entity classes.
    """
    all_pred = []
    all_ref = []

    for pred_labels, gold_labels in predictions:
        all_pred.extend(pred_labels)
        all_ref.extend(gold_labels)

    # Dynamically infer entity classes (exclude 'O' class 0)
    entity_classes = list(set(all_pred + all_ref) - {0})

    if not entity_classes:
        return 0.0  # No entities to evaluate

    # Compute F1 considering ALL tokens, but focusing on entity classes
    f1 = f1_score(
        all_ref,
        all_pred,
        average='micro',
        labels=entity_classes,
        zero_division=0
    )

    return f1
