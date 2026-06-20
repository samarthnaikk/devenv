def normalize_title(title: str) -> str:
    return " ".join(title.strip().split()).title()
