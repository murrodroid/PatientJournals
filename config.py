cfg = dict(
    model = 'gemini-3-pro-preview',
    verification_model = '',
    batch_size = 2048,
    dataset_file_name = 'test_dataset'
)

image_settings = dict(
    max_dim = 3000,
    contrast_factor = 1.1,
    margins = (50, # left
               0,  # top
               0,  # right
               0,  # bottom
               ),
    output_format = "PNG",
)

prompts = dict(
    primary = 
    f"""
    Context:
    You are given a scanned page from a Danish hospital patient journal from the late 1800s.
    Your task is to extract data from the content on the page.

    The typical order of basic information is: name, age, occupation, address.

    Objective:
    Fill each column with the information found in the image.
    If a column cannot be determined, return an empty string for that position.

    Guidelines:
    - Examples are always written as "Examples: [example1,example2,example3]"
    - Use only what is visible in the image.
    - Do not infer or guess beyond the evidence on the page.
    - Preserve spellings exactly as written, even if archaic or non-standard. Only exception is numbers, which should be written as float-values.
    - If nothing fits a Field, output an empty field for that position.
    """,
)