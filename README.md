# Mountain NER

Fine-tuned BERT (`bert-base-cased`) for named entity recognition of mountain names in English text.
Weights: [Oleks333/mountain-ner-bert](https://huggingface.co/Oleks333/mountain-ner-bert) on the Hugging Face Hub.
Labels follow the BIO scheme: `O`, `B-MOUNTAIN`, `I-MOUNTAIN`.

## Project structure

```
mountain-ner/
├── data/
│   ├── mountain_sentences.csv   # ~1000 sentences, one mountain each (scraped names + GPT sentences)
│   └── negative_sentences.csv   # ~200 sentences without mountains (empty `mountain`)
├── dataset_creation.ipynb       # scrapes mountain names and generates sentences with OpenAI
├── train_model.py               # fine-tuning + evaluation on a held-out test split
├── inference.py                 # CLI / Python API for predictions
├── demo.ipynb                   # usage examples
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Oleksandr190378/mountain-ner.git
cd mountain-ner
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`.env` with `OPENAI_API_KEY` is only needed to regenerate the dataset in `dataset_creation.ipynb`.

## Training

```bash
python train_model.py            # see --help for epochs, batch size, base model, etc.
```

The data is split 80/10/10 (train/validation/test, stratified by presence of a mountain).
The best checkpoint by validation F1 is saved to `model/mountain_ner_bert/` together with
`test_metrics.json`. Training takes ~7 minutes on a 2-core CPU; a GPU makes it a matter of a minute.

## Results

Entity-level metrics (`seqeval`) on the held-out test split:

| Split | Sentences | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation (best epoch) | 119 | 0.990 | 0.980 | 0.985 |
| Test | 120 | 0.990 | 1.000 | 0.995 |

3 epochs, batch size 16, learning rate 3e-5, `bert-base-cased`.

The test split comes from the same synthetic distribution as the training data, so these numbers
overestimate performance on real-world text. Known limitations seen on hand-written examples:

- every positive training sentence contains exactly one mountain, so the second mountain in a
  sentence is sometimes missed (e.g. "...looked at the Matterhorn");
- multi-word names not seen in training can be cut (e.g. "Fitz Roy" -> "Roy");
- short names such as "K2" get low confidence.

More varied data (several mountains per sentence, names mid-sentence) is the main way to improve it.

## Inference

No training needed: without a local `model/mountain_ner_bert/` the Hub model is downloaded automatically.

```bash
python inference.py "We climbed Mont Blanc and then looked at the Matterhorn."
```

```python
from inference import load_pipeline, predict_mountains

ner = load_pipeline()  # local model/mountain_ner_bert/ if present, else the Hub model
predict_mountains("Denali is the highest peak in North America.", ner)
# [{'mountain': 'Denali', 'score': 0.99, 'start': 0, 'end': 6}]
```

## License

MIT
