from trainer import trainer

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }

best_trials = trainer.hyperparameter_search(
    direction=["minimize", "maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    # compute_objective=compute_objective,
)
