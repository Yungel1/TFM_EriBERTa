from transformers import AutoConfig, AutoModelForTokenClassification


def define_model(model_name, label2id, id2label):

    # Model config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    config.max_position_embeddings = 512

    # Model definition
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

    return model
