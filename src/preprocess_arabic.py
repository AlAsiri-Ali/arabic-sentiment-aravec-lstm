from __future__ import annotations

import re
import string


ARABIC_PUNCTUATION = string.punctuation + "،؛؟ـ…"


def tokenize_arabic(text: str) -> list[str]:
    return str(text).split()


def clean_arabic_tokens(tokens: list[str]) -> list[str]:
    cleaned = []
    for token in tokens:
        token = re.sub(f"[{re.escape(ARABIC_PUNCTUATION)}]", " ", token)
        token = re.sub(r"\s+", " ", token).strip()
        if token:
            cleaned.append(token)
    return cleaned


def clean_arabic_text(text: str) -> str:
    return " ".join(clean_arabic_tokens(tokenize_arabic(text)))
