from __future__ import annotations
from openai import OpenAI

from .prompt import build_eval_prompt
from .parse_feedback import extract_level, extract_scores_block

def evaluate_transcription(client: OpenAI, transcription: str, model: str) -> tuple[str, dict]:
    prompt = build_eval_prompt(transcription)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un examinateur CECRL strict et fiable."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    feedback = (resp.choices[0].message.content or "").strip()
    level = extract_level(feedback)
    scores = extract_scores_block(feedback)
    return feedback, {"level": level, **scores}
