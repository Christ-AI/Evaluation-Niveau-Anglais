from __future__ import annotations

def build_eval_prompt(transcription: str) -> str:
    return f"""
Tu es un examinateur d'anglais (CECRL A1 à C2). Évalue le niveau d'un candidat à partir de sa transcription.

Transcription:
\"\"\"{transcription}\"\"\"

Donne une réponse STRICTEMENT au format suivant (respecte les titres) :

Niveau estimé: <A1/A2/B1/B2/C1/C2>

Scores (0-10):
- Grammaire:
- Vocabulaire:
- Prononciation (inférée depuis la transcription):
- Fluidité:

Forces:
- <bullet 1>
- <bullet 2>

Axes d'amélioration:
- <bullet 1>
- <bullet 2>

Recommandations concrètes:
- <bullet 1>
- <bullet 2>

Règles:
- Sois concis mais clair.
- Ne donne pas d'informations personnelles.
""".strip()
