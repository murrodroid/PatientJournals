from preprocess import preprocess_image
from google import genai
from google.genai import types
from config import *
from api_keys import gemini_maarten as api_key
import pandas as pd
from classes import Journal
from tools import data_to_row, flush_csv, create_subfolder, list_input_files
from generate import generate_data


def main():
    model = cfg.get('model')
    client = genai.Client(api_key=api_key)
    
    data = list_input_files(cfg)
    
    batch_size = cfg.get('batch_size',1024)
    rows: list[dict] = []
    header_written = False

    out_name = cfg.get('dataset_file_name', 'dataset')
    run_dir = create_subfolder(cfg.get("output_root", "runs"))
    out_csv_path = run_dir / f"{out_name}.csv"

    try:
        for file_name in data:
            journal_data = generate_data(client=client,model=model,file_name=file_name)
            journal_row = data_to_row(data=journal_data,file_name=file_name)
            
            rows.append(journal_row)

            if len(rows) >= batch_size:
                header_written = flush_csv(rows, str(out_csv_path), header_written)
                rows.clear() 

    except Exception as e:
        print(f"Stopping early due to error: {e}")

    finally:
        if rows:
            header_written = flush_csv(rows, str(out_csv_path), header_written)



if __name__ == "__main__":
    main()