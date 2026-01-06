import asyncio
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types
from tqdm.asyncio import tqdm_asyncio

from api_keys import gemini_maarten as api_key
from config import cfg, prompts, image_settings
from classes import Journal
from preprocess import preprocess_image
from tools import list_input_files, create_subfolder


UPLOAD_SEMAPHORE = asyncio.Semaphore(cfg.get('batch_upload_limit'))

async def upload_image_for_batch(client: genai.Client, file_path: str) -> str:
    """
    Preprocesses, saves to temp, uploads to GenAI File API, returns File URI.
    """
    async with UPLOAD_SEMAPHORE:
        image_bytes, mime_type = await asyncio.to_thread(
            preprocess_image,
            file_path,
            max_dim=image_settings.get("max_dim", 3000),
            margins=tuple(image_settings.get("margins", (0, 0, 0, 0))),
            contrast_factor=image_settings.get("contrast_factor", 1.0),
            output_format=image_settings.get("output_format", "PNG"),
        )

        # save to temp file
        original_stem = Path(file_path).stem
        
        with tempfile.NamedTemporaryFile(suffix=".png", prefix=f"{original_stem}_", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            # upload to Google GenAI File API
            uploaded_file = await client.aio.files.upload(path=tmp_path, config=dict(display_name=original_stem))
            return uploaded_file.name, uploaded_file.uri
        finally:
            # clean up local temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

async def prepare_batch_request(client: genai.Client, file_path: str, model: str):
    """Creates a single JSONL request entry"""
    try:
        # upload and get URI
        file_name_ref, file_uri = await upload_image_for_batch(client, file_path)
        
        # create the request object compatible with Batch API
        # store the original filename in 'custom_id' to map results back later
        request_id = Path(file_path).name 

        request = types.BatchJobRequest(
            request=types.GenerateContentRequest(
                model=model,
                contents=[
                    types.Part.from_uri(file_uri=file_uri, mime_type="image/png"),
                    types.Part.from_text(text=prompts["primary"])
                ],
                generation_config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Journal.model_json_schema(),
                )
            ),
            custom_id=request_id 
        )
        return request
    except Exception as e:
        print(f"Failed to prepare {file_path}: {e}")
        return None

async def main():
    client = genai.Client(api_key=api_key)
    model = cfg.get('model')
    
    files = list_input_files(cfg)
    print(f"Found {len(files)} files to process.")

    tasks = [prepare_batch_request(client, f, model) for f in files]
    batch_requests = []
    
    for coro in tqdm_asyncio.as_completed(tasks, desc="Uploading & Preparing"):
        req = await coro
        if req:
            batch_requests.append(req)

    if not batch_requests:
        print("No requests prepared. Exiting.")
        return

    # create the Batch Job
    # SDK handles the intermediate JSONL creation/upload internally usually, 
    # but strictly speaking we pass the list of requests.
    print(f"Submitting batch job with {len(batch_requests)} items...")
    
    batch_job = await client.aio.batches.create(
        model=model,
        src=batch_requests,
    )

    print(f"Batch Job Submitted Successfully!")
    print(f"Job Name: {batch_job.name}")
    print(f"Job State: {batch_job.state}")

    # save Metadata so we can retrieve it later
    run_dir = create_subfolder(cfg.get("output_root", "runs"))
    metadata = {
        "batch_job_name": batch_job.name,
        "submit_time": datetime.now().isoformat(),
        "file_count": len(batch_requests),
        "model": model
    }
    
    (run_dir / "batch_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved to {run_dir / 'batch_metadata.json'}")

if __name__ == "__main__":
    asyncio.run(main())
    # in terminal:
    # python batch_retrieve.py runs/20250501_103000
    # to specify run