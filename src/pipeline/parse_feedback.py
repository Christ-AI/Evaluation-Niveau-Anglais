from __future__ import annotations
import re

LEVEL_RE = re.compile(r"Niveau estimé:\s*(A1|A2|B1|B2|C1|C2)", re.IGNORECASE)

def extract_level(feedback: str) -> str:
    m = LEVEL_RE.search(feedback or "")
    return (m.group(1).upper() if m else "UNKNOWN")

def extract_scores_block(feedback: str) -> dict:
    # extraction simple, robuste
    def grab(label: str) -> float | None:
        r = re.search(rf"{re.escape(label)}\s*:\s*([0-9]+(\.[0-9]+)?)", feedback, re.IGNORECASE)
        return float(r.group(1)) if r else None

    return {
        "grammar_score": grab("Grammaire"),
        "vocab_score": grab("Vocabulaire"),
        "pron_score": grab("Prononciation"),
        "fluency_score": grab("Fluidité"),
    }
