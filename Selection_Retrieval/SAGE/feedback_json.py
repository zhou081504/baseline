"""Strict parsing of model feedback, including JSON inside Markdown fences."""
import json
import re


def parse_feedback(text):
    valid = []
    for match in re.finditer(r'\{[^{}]*\}', text):
        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if (set(value) == {'score', 'adjustment'}
                and type(value['score']) is int and 1 <= value['score'] <= 10
                and type(value['adjustment']) is int and value['adjustment'] in (-1, 1)):
            valid.append(value)
    if not valid or any(value != valid[0] for value in valid):
        raise ValueError('Feedback requires unambiguous JSON with integer score 1..10 and adjustment -1 or 1')
    return valid[0]
