import logging

import src.config.conf as cf
import src.infra.loader_utils as lu
import src.core.col_selection as cs
import src.core.feature_engineering as fe
import src.core.preprocessing as pp
import src.config.logger_config as lc
import src.ml.lgbm_optimization as lo
import src.ml.lgbm_train_test as tt

from src.ml.optimization_config import OptimizationConfig

import optuna
import json
import pandas as pd

logger = logging.getLogger(__name__)

# region Config_vars

cfg = cf.load_config("CHALLENGE_01")
paths = cfg.get('PATHS', None)

# Experiment values
STUDY_NAME = cfg.get('STUDY_NAME', None)

SEEDS = cfg.get('SEEDS', None)

MONTH_TRAIN = cfg.get('MONTH_TRAIN', None)
MONTH_VALIDATION = cfg.get('MONTH_VALIDATION', None)
MONTH_TEST = cfg.get('MONTH_TEST', None)

GAIN_AMOUNT = cfg.get('GAIN')
COST_AMOUNT = cfg.get('COST')


BINARY_POSITIVES = cfg.get('BINARY_POSITIVES', None)

LGBM_N_TRIALS = cfg.get('LGBM_N_TRIALS', None)
LGBM_N_FOLDS = cfg.get('LGBM_N_FOLDS', None)
LGBM_N_BOOSTS = cfg.get('LGBM_N_BOOSTS', None)
LGBM_THRESHOLD = cfg.get('LGBM_THRESHOLD', None)

# Paths

## Logs
PATH_LOGS = paths.get('LOGS', None)

## Input
PATH_DATA = paths.get('INPUT_DATA', None)

## Output
PATH_LGBM_OPT = paths.get('OUTPUT_LGBM_OPTIMIZATION', None)
PATH_LGBM_OPT_BEST_PARAMS = paths.get('OUTPUT_LGBM_OPTIMIZATION_BEST_PARAMS', None)
PATH_LGBM_OPT_DB = paths.get('OUTPUT_LGBM_OPTIMIZATION_DB', None)

PATH_LGBM_MODEL = paths.get('OUTPUT_LGBM_MODEL', None)

PATH_PREDICTION = paths.get('OUTPUT_PREDICTION')

# endregion 


def main():
    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = lu.load_data(f"{PATH_DATA}competencia_01.csv", "csv")

    # 1. Columns selection
    cols_lag_delta_max_min_regl, cols_ratios = cs.col_selection(df)

    # 2. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "delta": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        # "minmax": {
        #     "columns": cols_lag_delta_max_min_regl
        # },
        "ratio": {
            "pairs": cols_ratios
        },
        # "linreg": {
        #     "columns": cols_lag_delta_max_min_regl,
        #     "window": 3
        # }
    })

    # 3. Preprocessing
    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_VALIDATION
    )

    # 4. Hyperparameters optimization

    opt_cfg = OptimizationConfig(
        n_trials=LGBM_N_TRIALS,
        name=STUDY_NAME,

        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        n_folds=LGBM_N_FOLDS,
        n_boosts=LGBM_N_BOOSTS,
        seeds=SEEDS,
        output_path=PATH_LGBM_OPT
    )
 
    study = lo.run_lgbm_optimization(X_train, y_train_binary, w_train, opt_cfg)

    # 5. Entrenamiento lgbm con la mejor iteración y mejores hiperparámetros

    best_iter = study.best_trial.user_attrs["best_iter"]
    best_params = study.best_trial.params

    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=STUDY_NAME,

        output_path=PATH_LGBM_MODEL,
        seeds=SEEDS

    )
    model_lgbm = tt.entrenamiento_lgbm(X_train , y_train_binary, w_train ,best_iter,best_params , tt_cfg)
    y_pred=tt.evaluacion_lgbm(X_test , y_test_binary ,model_lgbm)


    logger.info("Pipeline ENDED!")


def kaggle_prediction():
    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = lu.load_data(f"{PATH_DATA}competencia_01.csv", "csv")

    # 1. Columns selection
    cols_lag_delta_max_min_regl, cols_ratios = cs.col_selection(df)

    # 2. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "delta": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        # "minmax": {
        #     "columns": cols_lag_delta_max_min_regl
        # },
        "ratio": {
            "pairs": cols_ratios
        },
        # "linreg": {
        #     "columns": cols_lag_delta_max_min_regl,
        #     "window": 3
        # }
    })

    # 3. Preprocessing
    MONTH_TRAIN.append(MONTH_VALIDATION)

    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_TEST
    )

    # 4. Best hyperparams loading
    name_best_params_file = f"best_params_binary{STUDY_NAME}.json"
    storage_name = "sqlite:///" + PATH_LGBM_OPT_DB + "optimization_lgbm.db"
    study = optuna.load_study(study_name='study_lgbm_binary'+STUDY_NAME, storage=storage_name)
    
    # 5. Training with best attempt and hyperparams
    best_iter = study.best_trial.user_attrs["best_iter"]
    
    with open(PATH_LGBM_OPT_BEST_PARAMS + name_best_params_file, "r") as f:
        best_params = json.load(f)
    logger.info(f"Hyperparams OK?: {study.best_trial.params == best_params}")
    
    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=STUDY_NAME,

        output_path=PATH_LGBM_MODEL,
        seeds=SEEDS
    )
    
    model_lgbm = tt.entrenamiento_lgbm(X_train, y_train_binary, w_train ,best_iter,best_params , tt_cfg)

    # 6. Prediction!

    y_test_binary=X_test[["numero_de_cliente"]]
    y_pred=model_lgbm.predict(X_test)
    y_test_binary["Predicted"] = y_pred
    y_test_binary["Predicted"]=y_test_binary["Predicted"].apply(lambda x : 1 if x >=0.025 else 0)
    logger.info(f"cantidad de bajas predichas : {(y_test_binary==1).sum()}")
    y_test_binary=y_test_binary.set_index("numero_de_cliente")
    y_test_binary.to_csv(f"output/prediction/prediccion{STUDY_NAME}.csv")

    logger.info("Pipeline ENDED!")

def compare():
    pred1 = pd.read_csv("output/prediction/prediccion_patito.csv", sep=',')
    pred2 = pd.read_csv("output/prediction/prediccion_20251003.csv", sep=',')

    merged = pred1.merge(pred2, on="numero_de_cliente", suffixes=("", "_patito"))
    diffs = merged[merged["Predicted"] != merged["Predicted_patito"]]

    print(f"Differences found: {len(diffs)}")
    if len(diffs) > 0:
        diffs.to_csv("output/diffs.csv", index=False)

if __name__ == "__main__":
    lu.ensure_dirs(
        PATH_LOGS,
        PATH_DATA,
        PATH_LGBM_OPT,
        PATH_LGBM_OPT_BEST_PARAMS,
        PATH_LGBM_OPT_DB,
        PATH_LGBM_MODEL,
        PATH_PREDICTION
    )
    lc.setup_logging(PATH_LOGS)

    main()
    kaggle_prediction()
    # compare()
