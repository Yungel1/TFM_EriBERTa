from transformers import AutoConfig, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate


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


def define_trainer(model, tokenizer, data_collator, train_dataset, eval_dataset,
                   compute_metrics, output_dir):

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        #metric_for_best_model="f1",
        #greater_is_better=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


class MetricsComputer:
    def __init__(self, id2label):
        self.id2label = id2label
        self.metric = evaluate.load("seqeval")

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

