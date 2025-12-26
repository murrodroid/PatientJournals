import asyncio
import json
import argparse
import sys
import pandas as pd
from pathlib import Path
from google import genai

from api_keys import gemini_maarten as api_key
from classes import Journal

def get_target_run_dir(user_path: str | None) -> Path | None:
    """
    Determines the run directory based on user input or auto-discovery.
    """
    # specified run
    if user_path:
        path = Path(user_path)
        if not path.exists():
            print(f"Error: The specified path '{user_path}' does not exist.")
            return None
        if not (path / "batch_metadata.json").exists():
            print(f"Error: No 'batch_metadata.json' found in '{user_path}'. Is this a batch run folder?")
            return None
        return path

    # latest run
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("Error: 'runs/' directory not found.")
        return None
        
    all_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    if not all_runs:
        print("Error: No run subfolders found in 'runs/'.")
        return None

    latest_run = all_runs[0]
    if not (latest_run / "batch_metadata.json").exists():
        print(f"Warning: The latest folder '{latest_run.name}' doesn't look like a batch run (missing metadata).")
    
    return latest_run

async def main():
    parser = argparse.ArgumentParser(description="Retrieve results from a Gemini Batch Job.")
    parser.add_argument(
        "run_folder", 
        nargs="?", 
        help="Path to the specific run folder (e.g., runs/20231226_120000). Defaults to latest if omitted."
    )
    args = parser.parse_args()

    target_run = get_target_run_dir(args.run_folder)
    if not target_run:
        sys.exit(1)

    print(f"Checking run: {target_run}")

    meta_path = target_run / "batch_metadata.json"
    metadata = json.loads(meta_path.read_text())
    job_name = metadata.get("batch_job_name")
    
    print(f"Job Name: {job_name}")

    client = genai.Client(api_key=api_key)
    try:
        job = await client.aio.batches.get(name=job_name)
    except Exception as e:
        print(f"Error fetching job status: {e}")
        return

    print(f"Current Status: {job.state}")

    if job.state == "COMPLETED":
        print("Job Complete. Downloading results...")
        
        rows = []
        async for item in await job.output_iterator():
            custom_id = item.custom_id
            
            if item.result.response.parts:
                response_text = item.result.response.parts[0].text
                try:
                    journal = Journal.model_validate_json(response_text)
                    
                    row = journal.model_dump(mode="python")
                    row["file_name"] = custom_id 
                    rows.append(row)
                except Exception as e:
                    print(f"Error parsing result for {custom_id}: {e}")
            else:
                 print(f"Empty response for {custom_id}")

        if rows:
            out_csv = target_run / "dataset_batch.csv"
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False, sep='$') 
            print(f"SUCCESS: Saved {len(rows)} rows to {out_csv}")
        else:
            print("Job completed but no valid rows were extracted.")

    elif job.state == "FAILED":
        print(f"Job Failed. Error: {job.error.message}")
    
    elif job.state == "CANCELLED":
        print("Job was cancelled.")

    else:
        print("Job is still processing. Please check again later.")

if __name__ == "__main__":
    asyncio.run(main())