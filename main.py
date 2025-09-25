import logging

import src.feature_engineering as fe
import src.col_selection as cs
from src.config.logger import setup_logging
from src.loader import load_data
from src.config.conf import load_config

logger = logging.getLogger(__name__)

def main():
    logger.info("STARTING this wonderful pipeline!")

    # region Config_vars
    cfg = load_config("competencia01")

    data_path = cfg.get('DATA_PATH', None)
    # endregion 

    # 0. Load data
    df = load_data(data_path, "csv")

    cols = cs.col_selection(df)

    # 1. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols[0],
            "n": 2   # number of lags
        },
        "delta": {
            "columns": cols[0],
            "n": 2   # number of deltas
        },
        "minmax": {
            "columns": cols[0]
        },
        "ratio": {
            "pairs": cols[1]
        },
        "linreg": {
            "columns": cols[0],
            "window": 3  # optional, for flexibility
        }
    })

    # df = fe.add_lag_features(df, cols[0], n=2)

    logger.info("Pipeline ENDED!")


if __name__ == "__main__":
    setup_logging()
    main()
