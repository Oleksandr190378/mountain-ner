import argparse
from pathlib import Path

from transformers import pipeline

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model" / "mountain_ner_bert"


def load_pipeline(model=DEFAULT_MODEL_DIR):
    return pipeline("token-classification", model=str(model), aggregation_strategy="first")


def predict_mountains(text, ner=None):
    ner = ner or load_pipeline()
    return [
        {"mountain": text[e["start"]:e["end"]], "score": round(float(e["score"]), 3), "start": e["start"], "end": e["end"]}
        for e in ner(text)
        if e["entity_group"] == "MOUNTAIN"
    ]


def main():
    parser = argparse.ArgumentParser(description="Find mountain names in text.")
    parser.add_argument("text", nargs="*", help="Sentence(s) to analyze")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Local model directory or Hugging Face Hub id")
    args = parser.parse_args()

    ner = load_pipeline(args.model)
    texts = args.text or ["I climbed Donguzorun last summer and it was an amazing experience."]
    for text in texts:
        print(text)
        print(predict_mountains(text, ner))


if __name__ == "__main__":
    main()
