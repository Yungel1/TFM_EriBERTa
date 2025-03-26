import torch
from sklearn.metrics import f1_score, precision_score, recall_score

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


def _aggreg_ner(predictions):
    pred, ref = zip(*predictions)
    # concat all the predictions and references
    all_pred = []
    for p in pred:
        all_pred.extend(p)
    all_ref = []
    for r in ref:
        all_ref.extend(r)
    # compute the F1 score
    f1 = f1_score(all_ref, all_pred, average=None)
    if len(f1) > 1:
        f1_sum = sum(f1[:-1]) / (len(f1) - 1)
    else:
        f1_sum = f1[0]

    return f1_sum
