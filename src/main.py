from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from src.pipeline.transcribe import transcribe_audio
from src.pipeline.evaluate import evaluate_transcription

def iter_audio_files(input_dir: str):
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Dossier introuvable: {p.resolve()}")
    for ext in ("*.mp3", "*.wav", "*.m4a"):
        for f in p.glob(ext):
            yield f

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Évaluation niveau d'anglais (audio -> transcription -> CECRL).")
    parser.add_argument("--input_dir", default=os.getenv("INPUT_DIR", "data/mp3"))
    parser.add_argument("--out_csv", default=os.getenv("OUTPUT_CSV", "data/results/results.csv"))
    parser.add_argument("--llm_model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--stt_model", default="gpt-4o-mini-transcribe")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant. Crée un .env à partir de .env.example et renseigne la clé.")

    client = OpenAI(api_key=api_key)

    files = list(iter_audio_files(args.input_dir))
    if not files:
        print(f"Aucun fichier audio trouvé dans: {args.input_dir}")
        return

    rows = []
    for f in tqdm(files, desc="Évaluation"):
        try:
            transcription = transcribe_audio(client, str(f), model=args.stt_model)
            feedback, parsed = evaluate_transcription(client, transcription, model=args.llm_model)

            rows.append({
                "file_name": f.name,
                "transcription": transcription,
                "gpt_feedback": feedback,
                **parsed
            })
        except Exception as e:
            rows.append({
                "file_name": f.name,
                "transcription": "",
                "gpt_feedback": f"ERROR: {e}",
                "level": "ERROR",
                "grammar_score": None,
                "vocab_score": None,
                "pron_score": None,
                "fluency_score": None,
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Résultats exportés: {out_path.resolve()}")

if __name__ == "__main__":
    main()
