import json
import re
import string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def format_ref(reference):
    def format_step(step: str) -> str:
        return step.strip('\n').strip().replace('\n', '  ')
    return format_step(" ".join(reference))

def check_answer(answer: str) -> bool:
    uncertainty_keywords = ["can't determin", "unknown", "cannot determin"]
    if any(keyword.casefold() in answer.casefold() for keyword in uncertainty_keywords):
        return False

    word_limit = 10
    if len(answer.split()) > word_limit:
        return False
    return True
