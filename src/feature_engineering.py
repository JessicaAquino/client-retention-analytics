import pandas as pd
import numpy as np
import logging

from src.utils.duck_db_conn import run_duckdb_query

logger = logging.getLogger(__name__)

def add_lags(df:pd.DataFrame, col_list:list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_list : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Starting to add lag features. df shape: {df.shape}")

    # SQL query construction
    sql="SELECT *"
    for attr in col_list:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f",lag({attr},{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df"

    df = run_duckdb_query(df, sql)

    logger.info(f"Lag features added. df shape: {df.shape}")
    return df

def add_deltas(df:pd.DataFrame , col_list:list[str],cant_lag:int=1 ) -> pd.DataFrame:
    """
    Genera variables de delta para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_list : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    # Armado de la consulta SQL
    logger.info(f"Comienzo feature de delta.  df shape: {df.shape}")
    sql="SELECT *"
    for attr in col_list:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df"

    df = run_duckdb_query(df, sql)

    logger.info(f"ejecucion delta finalizada. df shape: {df.shape}")
    return df

def add_minmax(df:pd.DataFrame|np.ndarray , col_list:list[str]) -> pd.DataFrame|np.ndarray:
    """
    Genera variables de max y min para los atributos especificados por numero de cliente  utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_list : list
        Lista de atributos para los cuales generar min y max. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    logger.info(f"Comienzo feature max min. df shape: {df.shape}")
      
    sql="SELECT *"
    for attr in col_list:
        if attr in df.columns:
            sql+=f", MAX({attr}) OVER (PARTITION BY numero_de_cliente) as MAX_{attr}, MIN({attr}) OVER (PARTITION BY numero_de_cliente) as MIN_{attr}"
        else:
            print(f"El atributo {attr} no se encuentra en el df")
    
    sql+=" FROM df"

    df = run_duckdb_query(df, sql)

    logger.info(f"ejecucion max min finalizada. df shape: {df.shape}")
    return df

def add_ratios(df:pd.DataFrame|pd.Series, col_list:list[list[str]] )->pd.DataFrame:
    """
    Genera variables de ratio para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_list : list[list]
        Lista de pares de col_list de monto y cantidad relacionados para generar ratios. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ratios agregadas"""
    logger.info(f"Comienzo feature ratio. df shape: {df.shape}")
    sql="SELECT *"
    for par in col_list:
        if par[0] in df.columns and par[1] in df.columns:
            sql+=f", if({par[1]}=0 ,0,{par[0]}/{par[1]}) as ratio_{par[0]}_{par[1]}"
        else:
            print(f"no se encontro el par de atributos {par}")

    sql+=" FROM df"

    df = run_duckdb_query(df, sql)

    logger.info(f"ejecucion ratio finalizada. df shape: {df.shape}")
    return df

def add_linreg_slope(df : pd.DataFrame|np.ndarray , col_list:list[str]) ->pd.DataFrame|np.ndarray:
    logger.info(f"Comienzo feature reg lineal. df shape: {df.shape}")
    sql="SELECT *"
    try:

        for attr in col_list:
            if attr in df.columns:
                sql+=f", regr_slope({attr} , cliente_antiguedad ) over ventana_3 as slope_{attr}"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=" FROM df window ventana_3 as (partition by numero_de_cliente order by foto_mes rows between 3 preceding and current row)"
    except Exception as e:
        logger.error(f"Error en la regresion lineal : {e}")
        raise
    
    df = run_duckdb_query(df, sql)

    logger.info(f"ejecucion reg lineal finalizada. df shape: {df.shape}")
    return df