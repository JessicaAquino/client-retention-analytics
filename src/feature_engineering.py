import polars as pl
import numpy as np
import logging

from src.utils.duck_db_conn import run_duckdb_query

logger = logging.getLogger(__name__)


def feature_engineering_pipeline(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Ejecuta el pipeline de feature engineering completo

    Parameters:
    -----------
    data_path : str
        Ruta al archivo de datos
    config : dict
        Configuración del pipeline. Ejemplo:

        "lag": {
            "columns": ["col1", "col2"],
            "n": 2   # number of lags
        },
        "delta": {
            "columns": ["col1", "col2"],
            "n": 2   # number of deltas
        },
        "minmax": {
            "columns": ["col1", "col2"]
        },
        "ratio": {
            "pairs": [["monto", "cantidad"], ["ingresos", "clientes"]]
        },
        "linreg": {
            "columns": ["col1"],
            "window": 3  # optional, for flexibility
        }

    Returns:
    --------
    pl.DataFrame
        DataFrame con las nuevas features agregadas
    """

    sql = "SELECT *"

    logger.info(f"Starting feature engineering pipeline. df shape: {df.shape}")

    if "lag" in config:
        for col in config["lag"]["columns"]:
            for i in range(1, config["lag"]["n"] + 1):
                sql += f", lag({col}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col}_lag_{i}"

    if "delta" in config:
        for col in config["delta"]["columns"]:
            for i in range(1, config["delta"]["n"] + 1):
                sql += f", {col} - {col}_lag_{i} AS delta_{i}_{col}"

    if "minmax" in config:
        for col in config["minmax"]["columns"]:
            sql += f", MAX({col}) OVER (PARTITION BY numero_de_cliente) AS MAX_{col}, MIN({col}) OVER (PARTITION BY numero_de_cliente) AS MIN_{col}"
    
    if "ratio" in config:
        for pair in config["ratio"]["pairs"]:
            sql += f", IF({pair[1]} = 0, 0, {pair[0]} / {pair[1]}) AS ratio_{pair[0]}_{pair[1]}"

    window_clause = ""
    if "linreg" in config:
        window_size = config["linreg"].get("window", 3)
        for col in config["linreg"]["columns"]:
            sql += f", REGR_SLOPE({col}, cliente_antiguedad) OVER ventana_{window_size} AS slope_{col}"
        window_clause = f" WINDOW ventana_{window_size} AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW)"

    sql += " FROM df"
    if window_clause != "":
        sql += window_clause

    df = run_duckdb_query(df, sql)

    logger.info(f"Feature engineering pipeline completed. df shape: {df.shape}")

    return df
