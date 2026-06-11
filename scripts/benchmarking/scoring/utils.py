import re


def extract_ans(output: str | None) -> str | None:
    """Return the text between the first <ANS>...</ANS> pair, stripped. None if missing."""
    if output is None:
        return None
    m = re.search(r"<ANS>(.*?)</ANS>", output, re.DOTALL)
    return m.group(1).strip() if m else None
