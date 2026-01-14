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
    total_written = 0

    out_name = cfg.get('dataset_file_name', 'dataset')
    output_format = cfg.get('output_format', 'csv')
    run_dir = create_subfolder(cfg.get('output_root', 'runs'))
    log = get_run_logger(run_dir)
    out_path = run_dir / f'{out_name}.{output_format.lstrip(".")}'

    sem = asyncio.Semaphore(cfg.get('concurrent_tasks'))

    try:
        log(f"Starting run. Files={len(data)} Output={out_path.name}")
        tasks = [process_file(sem, client, model, f, log) for f in data]

        for coro in tqdm_asyncio.as_completed(tasks, desc='Processing images', unit='img'):
            journal_row = await coro
            
            if journal_row:
                rows.append(journal_row)
            else:
                log("Received empty row from processing step.")

            if len(rows) >= batch_size:
                header_written = flush_rows(
                    rows=rows,
                    out_path=str(out_path),
                    header_written=header_written,
                    output_format=output_format,
                )
                total_written += len(rows)
                rows.clear() 

    except Exception as e:
        write_run_error(run_dir, e)
        log("Stopping early due to error.", exc=e)
        print(f'Stopping early due to error: {e}')

    finally:
        if rows:
            header_written = flush_rows(
                rows=rows,
                out_path=str(out_path),
                header_written=header_written,
                output_format=output_format,
            )
            total_written += len(rows)
            log(f"Wrote final batch of {len(rows)} rows.")
        else:
            if total_written == 0:
                log("No rows written; output file may be missing.")


if __name__ == '__main__':
    asyncio.run(main())
