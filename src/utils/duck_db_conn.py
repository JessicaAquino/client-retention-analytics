import duckdb
import pandas as pd

def run_duckdb_query(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Executes a DuckDB SQL query over a DataFrame and returns the result."""
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        return con.execute(sql).df()
    finally:
        con.close()
