import logging

import src.col_selection as cs
import src.feature_engineering as fe
import src.preprocessing as pp
from src.config.logger import setup_logging
from src.loader import load_data
from src.config.conf import load_config

logger = logging.getLogger(__name__)

def main():
    logger.info("STARTING this wonderful pipeline!")

    # region Config_vars
    cfg = load_config("challenge01")

    data_path = cfg.get('PATH_INPUT_DATA', None)
    month_train = cfg.get('MONTH_TRAIN', None)
    month_validation = cfg.get('MONTH_VALIDATION', None)
    month_test = cfg.get('MONTH_TEST', None)
    
    lgbm_opt_path = cfg.get('PATH_OUTPUT_LGBM_OPTIMIZATION', None)
    lgbm_model_path = cfg.get('PATH_OUTPUT_LGBM_MODEL', None)
    
    gain_amount = cfg.get('GAIN')
    cost_amount = cfg.get('COST')

    # endregion 

    # 0. Load data
    df = load_data(data_path, "csv")

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
        ["BAJA+2", "BAJA+1"],
        month_train,
        month_validation
    )

    print(X_train)

    logger.info("Pipeline ENDED!")


if __name__ == "__main__":
    setup_logging()
    main()
