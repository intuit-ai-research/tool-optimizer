def str_is_empty(string: str) -> bool:
    return string is None or string.strip() == ""


def str_is_not_empty(string: str) -> bool:
    return not str_is_empty(string)
