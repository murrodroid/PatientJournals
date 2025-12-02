
from tools import render_columns

cfg = dict(
    columns = dict(
        name = "type:str, The name of the patient.",
        age = "type:int>0, The age of the patient.",
        diagnosesObs = "type:str, The diagnosis as written by the physician.",
        dateOfAdmission = "type:datetime, The day the patient was admitted to the hospital.",
        physician = "type:str, The physician who did the survey.",
    ),
    model = 'gemini-2.5-flash-preview-09-2025',
    verification_model = '',
)

image_settings = dict(
    max_dim = 3000,
    contrast_factor = 1.1,
    output_format = "PNG",
)

prompts = dict(
    primary = f"""
    Context:
    You are given a scanned page from a Danish hospital patient journal from the late 1800s.
    Your task is to extract data from the content on the page.

    Objective:
    Fill each column with the information found in the image.
    If a column cannot be determined, return an empty string for that position.

    Output format:
    Return a single dollar-separated line, using the character $ as the separator.
    Each value must correspond to its column in the exact order listed below.
    Return nothing except this single dollar-separated line.
    Do not add spaces before or after the $ separators.

    Columns:
    {render_columns(cfg.get('columns'))}

    Guidelines:
    - Use only what is visible in the image.
    - Do not infer or guess beyond the evidence on the page.
    - Preserve spellings exactly as written, even if archaic or non-standard.
    - If multiple values exist for a field, choose the most prominent or clearly stated one.
    - If nothing fits a column, output an empty field for that position (e.g., value1$$value3 for three columns where the middle one is unknown).
    - Replace any newline characters within a field with a single space.
    - If the original text contains the $ character, replace it with the token [DOLLAR] in the output.
    """,
)