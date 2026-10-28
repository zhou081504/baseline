"""Lightweight sentence splitting shared by SAGE data collection and inference."""
import re


def sentences(text):
    return [s.strip() for s in re.findall(r'[^。！？!?\n]+[。！？!?]?|[^\n]+$', text) if s.strip()] or [text]
