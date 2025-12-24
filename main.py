import asyncio
from google import genai
from tqdm.asyncio import tqdm_asyncio

from api_keys import gemini_maarten as api_key
from config import *
from tools import *
from generate import process_file


async def main():
    model = cfg.get('model')
    client = genai.Client(api_key=api_key)
    
    data = list_input_files(cfg)
    
    batch_size = cfg.get('batch_size')
    rows: list[dict] = []
    header_written = False

    out_name = cfg.get('dataset_file_name', 'dataset')
    run_dir = create_subfolder(cfg.get("output_root", "runs"))
    out_csv_path = run_dir / f"{out_name}.csv"

    sem = asyncio.Semaphore(cfg.get('concurrent_tasks'))

    try:
        tasks = [process_file(sem, client, model, f) for f in data]

        for coro in tqdm_asyncio.as_completed(tasks, desc="Processing images", unit="img"):
            journal_row = await coro
            
            if journal_row:
                rows.append(journal_row)

            if len(rows) >= batch_size:
                header_written = flush_csv(rows=rows, out_csv=str(out_csv_path), header_written=header_written)
                rows.clear() 

    except Exception as e:
        write_run_error(run_dir, e)
        print(f"Stopping early due to error: {e}")

    finally:
        if rows:
            header_written = flush_csv(rows=rows, out_csv=str(out_csv_path), header_written=header_written)


if __name__ == "__main__":
    asyncio.run(main())