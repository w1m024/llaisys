THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def assistant_prefills_think(prompt_text: str) -> bool:
    return (prompt_text or "").rstrip().endswith(THINK_OPEN_TAG)


def normalize_assistant_text(text: str, prompt_prefilled_think: bool = False) -> str:
    normalized = text or ""
    close_index = normalized.find(THINK_CLOSE_TAG)
    open_index = normalized.find(THINK_OPEN_TAG)

    if prompt_prefilled_think and (open_index == -1 or open_index > close_index):
        prefix = THINK_OPEN_TAG if normalized.startswith("\n") else f"{THINK_OPEN_TAG}\n"
        return f"{prefix}{normalized}"

    if close_index != -1 and (open_index == -1 or open_index > close_index):
        prefix = THINK_OPEN_TAG if normalized.startswith("\n") else f"{THINK_OPEN_TAG}\n"
        return f"{prefix}{normalized}"

    return normalized
