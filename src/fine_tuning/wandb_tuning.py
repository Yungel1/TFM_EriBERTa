import wandb
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification


# TODO cambiar batch size (mi pc no da para más de 16, pendiente de cambiar)
def configure_sweep():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval/overall_f1", "goal": "maximize"},
        "parameters": {
            "batch_size": {"values": [8, 16, 32]},
            "learning_rate": {"values": [7.5e-5, 5e-5, 3e-5, 2e-5, 1e-5]},
            "weight_decay": {"values": [0.0, 0.01, 0.1, 0.2, 0.3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="eriberta-casimedicos_ner")
    return sweep_id


def train_model_wandb(model_name, model_config, device, tokenizer, data_collator, train_dataset,
                      eval_dataset, compute_metrics, output_dir):

    with wandb.init(config=None):

        model = AutoModelForTokenClassification.from_pretrained(model_name, config=model_config,
                                                                ignore_mismatched_sizes=True)
        model.to(device)

        config = wandb.config

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/wandb",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            num_train_epochs=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="overall_f1",
            greater_is_better=True,
            save_total_limit=3,
            report_to="wandb"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
