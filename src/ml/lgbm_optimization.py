from src.config.optimization_config import OptimizationConfig
from src.ml.lgbm_objective import LightGBMObjective
from src.ml.optuna_runner import OptunaRunner

import logging

logger = logging.getLogger(__name__)

def run_lgbm_optimization(X_train, y_train, w_train, cfg: OptimizationConfig):
    objective = LightGBMObjective(X_train, y_train, w_train, cfg)
    runner = OptunaRunner(cfg)

    study = runner.run_study(objective)
    runner.save_best_params(study)
    logger.info("Optimization completed successfully.")

    return study
