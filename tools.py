import pandas as pd
from classes import Journal
from config import cfg

def data_to_row(data: Journal, file_name: str) -> dict:
    row = data.model_dump(mode="python")
    row["file_name"] = file_name
    row["model_used"] = cfg.get('model',None)
    return row

def flush_csv(rows: list[dict], out_csv: str, header_written: bool, sep: str = '$') -> bool:
    pd.DataFrame.from_records(rows).to_csv(
        out_csv,
        mode="a",
        index=False,
        header=not header_written,
        sep=sep
    )
    return True