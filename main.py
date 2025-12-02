from preprocess import preprocess_image
from google import genai
from google.genai import types
from config import *
from api_keys import gemini
import pandas as pd
from tools import parse_dollar_separated

client = genai.Client(api_key=gemini)

data = ['test_image.png']

df = pd.DataFrame(columns=[key for key in cfg.get('columns').keys()] + ['file_name'])

try:
    for i, d in enumerate(data):
        image_bytes, mime_type = preprocess_image(
            d,
            max_dim=image_settings.get('max_dim'),
            contrast_factor=image_settings.get('contrast_factor'),
            output_format=image_settings.get('output_format'),
        )

        output_str = client.models.generate_content(
            model=cfg.get('model'),
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                ),
                prompts.get('primary', 'ERROR retrieving prompt. Return OUTPUT = ERROR.')
            ]
        )

        output_list = parse_dollar_separated(output_str)

        if len(output_list) == len(cfg.get('columns')):
            row = output_list + [d]
            df.loc[i] = row
        else:
            raise AssertionError('Length of output does not match columns.')
except Exception as e:
    print(f"Stopping early due to error: {e}")
finally:
    df.to_excel('test_dataset.xlsx')

if __name__ == "__main__":
    pass