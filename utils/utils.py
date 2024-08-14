from typing import Optional


def safe_concat(string: Optional[str], suffix: Optional[str]) -> str | None:
    match (string, suffix):
        case (None, None):
            return None
        case (s, None) | (None, s):
            return s
        case (s, t):
            return s + t
