

def render_columns(columns):
    items = [f"{key}: {value}" for key, value in columns.items()]
    return "\n".join(items)

def parse_separated(text,symbol='$'):
    return [field.strip() for field in text.split(symbol)]