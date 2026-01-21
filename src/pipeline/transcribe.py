from __future__ import annotations
from pathlib import Path
from openai import OpenAI

def transcribe_audio(client: OpenAI, audio_path: str, model: str = "gpt-4o-mini-transcribe") -> str:
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier audio introuvable: {p.resolve()}")

    with p.open("rb") as f:
        # API transcription (selon versions, ce modèle peut être ajusté)
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
        )

    # Certaines versions renvoient .text, d'autres un dict-like
    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
    return (text or "").strip()
