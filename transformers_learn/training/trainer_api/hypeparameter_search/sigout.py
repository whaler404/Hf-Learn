from trainer import trainer

def sigopt_hp_space(trial):
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
        {
            "categorical_values": ["16", "32", "64", "128"],
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
    ]

best_trials = trainer.hyperparameter_search( 
    direction=["minimize", "maximize"],
    backend="sigopt",
    hp_space=sigopt_hp_space,
    n_trials=20,
    # compute_objective=compute_objective,
)