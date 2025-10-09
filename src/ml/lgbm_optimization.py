from src.ml.lgbm_objective import LightGBMObjective
from src.ml.optuna_runner import OptunaRunner

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    n_trials: int
    name: str

    gain_amount: int
    cost_amount: int

    n_folds: int
    n_boosts: int
    seeds: list[int]
    output_path: str

def run_lgbm_optimization(X_train, y_train, w_train, cfg: OptimizationConfig):
    objective = LightGBMObjective(X_train, y_train, w_train, cfg)
    runner = OptunaRunner(cfg)

    study = runner.run_study(objective)
    runner.save_best_params(study)
    logger.info("Optimization completed successfully.")

    return study
