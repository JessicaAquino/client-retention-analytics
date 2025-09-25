import pandas as pd
import numpy as np
import os
import datetime
import logging

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

    # 1. Feature Engineering
    logger.info("Pipeline ENDED!")


if __name__ == "__main__":
    setup_logging()
    main()
