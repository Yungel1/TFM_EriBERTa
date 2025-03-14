from transformers import AutoConfig, AutoModelForTokenClassification, Trainer
import wandb
import optuna


class ModelInitializer:
    def __init__(self, model_name, label2id, id2label):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
        )

    def __call__(self):
        return AutoModelForTokenClassification.from_pretrained(self.model_name, config=self.config, ignore_mismatched_sizes=True)


def define_opt_trainer(training_args, model_init_instance, tokenizer, data_collator, train_dataset, eval_dataset, compute_metrics):

    return Trainer(
        model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        model_init=model_init_instance,
        data_collator=data_collator,
    )


def search_best_hyperparams(model_init_instance, tokenizer, data_collator, train_dataset, eval_dataset, compute_metrics, output_dir):
    # f1-score max, gemma, llama
    # Iniciar W&B para registrar los experimentos
    wandb.init(project="eriberta-ner-hyperparam-optimization", job_type="hyperparam_search")

    # Definir el espacio de búsqueda
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
            "weight_decay": trial.suggest_uniform("weight_decay", 0.0, 0.1),
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }

    trainer = define_opt_trainer(model_init_instance, tokenizer, data_collator, train_dataset, eval_dataset, compute_metrics)

    # Usar Trainer para la búsqueda
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10,
    )

    wandb.finish()

    print("Mejores hiperparámetros encontrados:", best_trial.hyperparameters)
    return best_trial.hyperparameters
