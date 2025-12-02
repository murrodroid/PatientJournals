

def render_columns(columns):
    items = [f"{key}: {value}" for key, value in columns.items()]
    return "\n".join(items)

def parse_dollar_separated(text):
    return [field.strip() for field in text.split("$")]