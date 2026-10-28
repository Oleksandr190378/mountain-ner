import argparse
from pathlib import Path

from transformers import pipeline

HUB_MODEL_ID = "Oleks333/mountain-ner-bert"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model" / "mountain_ner_bert"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def resolve_model():
    """Local trained model if it has weights, otherwise the Hugging Face Hub model."""
    has_weights = any((LOCAL_MODEL_DIR / f).exists() for f in WEIGHT_FILES)
    return LOCAL_MODEL_DIR if has_weights else HUB_MODEL_ID


def load_pipeline(model=None):
    model = model or resolve_model()
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
    parser.add_argument("--model", help=f"Local model directory or Hub id (default: model/mountain_ner_bert/ if trained, else {HUB_MODEL_ID})")
    args = parser.parse_args()

    ner = load_pipeline(args.model)
    texts = args.text or ["I climbed Donguzorun last summer and it was an amazing experience."]
    for text in texts:
        print(text)
        print(predict_mountains(text, ner))


if __name__ == "__main__":
    main()
