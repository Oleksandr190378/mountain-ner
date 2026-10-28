import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
DATA_FILES = [ROOT / "data" / "mountain_sentences.csv", ROOT / "data" / "negative_sentences.csv"]
DEFAULT_MODEL_DIR = ROOT / "model" / "mountain_ner_bert"

LABELS = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100


def load_data(files=DATA_FILES):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["mountain"] = df["mountain"].fillna("").str.strip()
    df["sentence"] = df["sentence"].str.strip()
    has_span = df.apply(lambda r: r.mountain == "" or r.mountain.lower() in r.sentence.lower(), axis=1)
    skipped = (~has_span).sum()
    if skipped:
        print(f"Skipping {skipped} rows where the mountain name is not found in the sentence")
    return df[has_span].reset_index(drop=True)


def find_span(sentence, mountain):
    if not mountain:
        return None
    start = sentence.lower().find(mountain.lower())
    return (start, start + len(mountain))


def encode(df, tokenizer, max_length):
    encodings = tokenizer(
        df["sentence"].tolist(), truncation=True, max_length=max_length, return_offsets_mapping=True
    )
    examples = []
    for i, (sentence, mountain) in enumerate(zip(df["sentence"], df["mountain"])):
        span = find_span(sentence, mountain)
        labels, inside = [], False
        for start, end in encodings["offset_mapping"][i]:
            if start == end:
                labels.append(IGNORE_INDEX)
                continue
            if span and start < span[1] and end > span[0]:
                labels.append(LABEL2ID["I-MOUNTAIN" if inside else "B-MOUNTAIN"])
                inside = True
            else:
                labels.append(LABEL2ID["O"])
                inside = False
        examples.append(
            {
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
                "labels": labels,
            }
        )
    return examples


def to_label_sequences(predictions, label_ids):
    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(predictions, label_ids):
        keep = label_row != IGNORE_INDEX
        true_seqs.append([ID2LABEL[i] for i in label_row[keep]])
        pred_seqs.append([ID2LABEL[i] for i in pred_row[keep]])
    return true_seqs, pred_seqs


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    true_seqs, pred_seqs = to_label_sequences(np.argmax(logits, axis=-1), label_ids)
    return {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall": recall_score(true_seqs, pred_seqs),
        "f1": f1_score(true_seqs, pred_seqs),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT to recognize mountain names (NER).")
    parser.add_argument("--model-name", default="bert-base-cased")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    df = load_data()
    has_mountain = df["mountain"] != ""
    train_df, rest_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=has_mountain)
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=args.seed, stratify=rest_df["mountain"] != ""
    )
    print(f"Train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    train_data, val_data, test_data = (
        encode(split, tokenizer, args.max_length) for split in (train_df, val_df, test_df)
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir.parent / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=math.ceil(len(train_data) / args.batch_size) * args.epochs // 10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        seed=args.seed,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    output = trainer.predict(test_data)
    true_seqs, pred_seqs = to_label_sequences(np.argmax(output.predictions, axis=-1), output.label_ids)
    print("\nTest set (entity-level):")
    print(classification_report(true_seqs, pred_seqs, digits=3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metrics = {k.removeprefix("test_"): v for k, v in output.metrics.items()}
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
