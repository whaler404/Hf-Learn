from trainer import trainer
from ray import tune

def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }

best_trials = trainer.hyperparameter_search( 
    direction=["minimize", "maximize"],
    backend="ray",
    hp_space=ray_hp_space,
    n_trials=20,
    # compute_objective=compute_objective,
)