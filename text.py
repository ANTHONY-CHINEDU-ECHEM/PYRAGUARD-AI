"""Text normalisation shared by the lexical index, the hashing embedder and the grounding check."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    may might must of on or shall should so such than that the their them then there these they this those to
    was were what when where which who why will with would you your not no nor only own same too very s t just
    about above after again all also any because before being below between both during each few further here
    more most other our out over some through under until up while we us he she his her itself""".split()
)

# domain synonyms folded onto one canonical token so that "blaze" still finds "fire"
SYNONYMS = {
    "flame": "fire", "flames": "fire", "blaze": "fire", "burning": "fire", "alight": "fire",
    "fumes": "smoke", "haze": "smoke", "smouldering": "smoke", "smoulder": "smoke", "smoky": "smoke",
    "evacuate": "evacuation", "evacuating": "evacuation", "evacuated": "evacuation", "escape": "evacuation",
    "extinguishers": "extinguisher", "extinguish": "extinguisher",
    "batteries": "battery", "li": "lithium",
    "fat": "oil", "fats": "oil", "oils": "oil", "fryer": "oil", "pan": "oil",
    "servers": "server", "datacentre": "server", "datacenter": "server",
    "overheated": "overheating", "overheat": "overheating", "hotspot": "overheating", "hot": "overheating",
    "cylinders": "cylinder", "lpg": "gas", "propane": "gas",
    "wardens": "warden", "marshal": "warden", "marshals": "warden",
    "disabled": "assistance", "wheelchair": "assistance", "mobility": "assistance", "peep": "assistance",
    "co2": "carbon", "dioxide": "carbon",
    "racking": "warehouse", "pallets": "warehouse", "pallet": "warehouse",
    "legal": "law", "legislation": "law", "regulations": "law", "duty": "law", "duties": "law",
}


def stem(token: str) -> str:
    """Very light suffix stripping. Enough to merge plurals and common verb forms."""
    if token in SYNONYMS:
        return SYNONYMS[token]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    # operate, operated and operating must meet: drop a final e from longer words
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = _WORD.findall(text.lower())
    out = []
    for tok in tokens:
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(SYNONYMS.get(stem(tok), stem(tok)))
    return out


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text.replace("\n", " ")) if p.strip()]
    return parts
