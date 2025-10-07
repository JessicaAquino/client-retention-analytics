import logging

import src.config.conf as cf
import src.infra.loader as ld
import src.core.col_selection as cs
import src.core.feature_engineering as fe
import src.core.preprocessing as pp
import src.config.logger_config as lc
import src.ml.lgbm_optimization as lo
import src.ml.lgbm_train_test as tt

import optuna
import json
import pandas as pd

logger = logging.getLogger(__name__)

# region Config_vars

execution_name = "_20251006_2"

cfg = cf.load_config("challenge01")

DATA_PATH = cfg.get('PATH_INPUT_DATA', None)

MONTH_TRAIN = cfg.get('MONTH_TRAIN', None)
MONTH_VALIDATION = cfg.get('MONTH_VALIDATION', None)
MONTH_TEST = cfg.get('MONTH_TEST', None)

LGBM_OPT_PATH = cfg.get('PATH_OUTPUT_LGBM_OPTIMIZATION', None)
LGBM_MODEL_PATH = cfg.get('PATH_OUTPUT_LGBM_MODEL', None)

GAIN_AMOUNT = cfg.get('GAIN')
COST_AMOUNT = cfg.get('COST')

SEEDS = cfg.get('SEEDS', None)

LGBM_N_FOLDS = cfg.get('LGBM_N_FOLDS', None)
LGBM_N_BOOSTS = cfg.get('LGBM_N_BOOSTS', None)
LGBM_N_TRIALS = cfg.get('LGBM_N_TRIALS', None)

BINARY_POSITIVES = cfg.get('BINARY_POSITIVES', None)

# endregion 


def main():
    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = ld.load_data(DATA_PATH, "csv")

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
        "linreg": {
            "columns": cols_lag_delta_max_min_regl,
            "window": 3
        }
    })

    # 3. Preprocessing
    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_VALIDATION
    )

    # 4. Hyperparameters optimization
    name_lgbm=execution_name

    opt_cfg = lo.OptimizationConfig(
        n_trials=LGBM_N_TRIALS,
        name=name_lgbm,

        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        n_folds=LGBM_N_FOLDS,
        n_boosts=LGBM_N_BOOSTS,
        seeds=SEEDS,
        output_path=LGBM_OPT_PATH
    )
 
    study = lo.run_lgbm_optimization(X_train, y_train_binary, w_train, opt_cfg)

    # 5. Entrenamiento lgbm con la mejor iteración y mejores hiperparámetros

    best_iter = study.best_trial.user_attrs["best_iter"]
    best_params = study.best_trial.params

    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=name_lgbm,

        output_path=LGBM_MODEL_PATH,
        seeds=SEEDS

    )
    model_lgbm = tt.entrenamiento_lgbm(X_train , y_train_binary, w_train ,best_iter,best_params , tt_cfg)
    y_pred=tt.evaluacion_lgbm(X_test , y_test_binary ,model_lgbm)


    logger.info("Pipeline ENDED!")


def kaggle_prediction():
    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = ld.load_data(DATA_PATH, "csv")

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
        "linreg": {
            "columns": cols_lag_delta_max_min_regl,
            "window": 3
        }
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
    name_lgbm=execution_name
    name_best_params_file=f"best_params_binary{name_lgbm}.json"
    storage_name = "sqlite:///" + LGBM_OPT_PATH + "db/" + "optimization_lgbm.db" # Refactor
    study = optuna.load_study(study_name='study_lgbm_binary'+name_lgbm,storage=storage_name)
    
    # 5. Training with best attempt and hyperparams
    best_iter = study.best_trial.user_attrs["best_iter"]
    
    with open(LGBM_OPT_PATH + "best_params/"+name_best_params_file, "r") as f:
        best_params = json.load(f)
    logger.info(f"Hyperparams OK?: {study.best_trial.params == best_params}")
    
    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=name_lgbm,

        output_path=LGBM_MODEL_PATH,
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
    y_test_binary.to_csv(f"output/prediction/prediccion{name_lgbm}.csv")

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
    lc.setup_logging()
    main()
    kaggle_prediction()
    # compare()
