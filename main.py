from preprocess import preprocess_image
from google import genai
from google.genai import types
from config import *
from api_keys import gemini as api_key
import pandas as pd
from tools import parse_separated

def main():
    client = genai.Client(api_key=api_key)

    data = ['test_image.png']

    df = pd.DataFrame(columns=[key for key in columns.keys()] + ['file_name'])

    try:
        for i, d in enumerate(data):
            image_bytes, mime_type = preprocess_image(
                d,
                max_dim=image_settings.get('max_dim'),
                contrast_factor=image_settings.get('contrast_factor'),
                output_format=image_settings.get('output_format'),
            )

            output = client.models.generate_content(
                model=cfg.get('model'),
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    ),
                    prompts.get('primary')
                ]
            )

            output_list = parse_separated(output.text, symbol='$')

            if len(output_list) == len(columns):
                row = output_list + [d]
                df.loc[i] = row
            else:
                raise AssertionError('Length of output does not match columns.')
        
            # add verification loop using other model/prompt    

    except Exception as e:
        print(f"Stopping early due to error: {e}")
    finally:
        df.to_excel('test_dataset.xlsx')


if __name__ == "__main__":
    main()