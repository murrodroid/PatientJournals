cfg = dict(
    model = 'gemini-3-pro-preview',
    concurrent_tasks = 6,
    verification_model = '',
    batch_size = 2048,
    dataset_file_name = 'dataset',
    target_folder='data/test',
    input_glob='*.png',
    recursive=False,
    output_format = 'jsonl',
    output_root='runs',
    batch_upload_limit = 20,
)

image_settings = dict(
    max_dim = 3000,
    contrast_factor = 1.1,
    margins = (300, # left
               0,  # top
               0,  # right
               0,  # bottom
               ),
    output_format = 'PNG',
)

prompts = dict(
    primary = 
    f"""
    Context:
    You are given a scanned page from a Danish hospital patient journal from the late 1800s.
    Your task is to extract data from the content on the page.

    Objective:
    Fill each column with the information found in the image.
    Not all columns are present within an image, meaning it isn't necessary to fill out all.

    Guidelines:
    - Examples are always written as 'Examples: [example1,example2,example3]'
    - Use only what is visible in the image.
    - Do not infer or guess beyond the evidence on the page.
    - Preserve spellings exactly as written, even if archaic or non-standard. Only exception is numbers, which should be written as float-values.
    - If nothing fits a Field, output an empty field for that position.
    """,
)