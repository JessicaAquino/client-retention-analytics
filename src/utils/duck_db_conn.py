import duckdb
import polars as pl

def run_duckdb_query(df: pl.DataFrame, sql: str) -> pl.DataFrame:
    """Executes a DuckDB SQL query over a DataFrame and returns the result."""
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        return con.execute(sql).df()
    finally:
        con.close()
