from google import genai
from google.genai import types
from preprocess import preprocess_image
from config import *
from classes import Journal


def generate_data(client: genai.Client, model: str, file_name: str) -> Journal:
    image_bytes, mime_type = preprocess_image(
        file_name,
        max_dim=image_settings.get("max_dim", 3000),
        margins=tuple(image_settings.get("margins", (0, 0, 0, 0))),
        contrast_factor=image_settings.get("contrast_factor", 1.0),
        output_format=image_settings.get("output_format", "PNG"),
    )

    output = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompts.get("primary"),
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Journal.model_json_schema(),
        },
    )

    return Journal.model_validate_json(output.text)