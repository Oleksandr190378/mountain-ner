# Mountain Name Recognition

This project aims to fine-tune a BERT model for Named Entity Recognition (NER) to detect mountain names in text. The model is trained on a dataset of sentences containing mountain names and is capable of predicting the names of mountains in new sentences.

## Project Structure

```
mountain-ner/
├── .env                  # OPENAI_API_KEY (only for dataset_creation.ipynb)
├── dataset_creation.ipynb
├── train_model.py
├── inference.py
├── demo.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── mountain_sentences.csv

```

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Oleksandr190378/mountain-ner.git
   cd mountain-ner
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables in `.env`**.

## Usage

### Option 1: Using the Pre-trained Model

If you want to use the pre-trained model and tokenizer, follow these steps:

1. **Load the pre-trained model and tokenizer**:
   - Link: Model Weights on Google Drive
   - Download and place  in the model/mountain_ner_bert directory.

2. **Make predictions**:
   - Use the `inference.py` script to make predictions on new sentences.

3. **Interactive demo**:
   - Run `demo.ipynb` for an interactive demonstration of the model's capabilities.

### Option 2: Training a Custom Model

If you want to create a custom dataset and train your own model, follow these steps:

1. **Create dataset**:
   - Run `dataset_creation.ipynb` to generate a dataset of sentences containing mountain names.
   - The dataset will be saved to `data/mountain_sentences.csv`.

2. **Train model**:
   - Execute `python train_model.py` to train the BERT model on your custom dataset.
   - The trained model and tokenizer will be saved to `model/mountain_ner_bert/`.

3. **Make predictions**:
   - Use the `inference.py` script to make predictions on new sentences using your custom model.

## Model

We use a fine-tuned BERT model for token classification to identify mountain names in text. The model and tokenizer are saved in the `model/` directory after training.

## Data

The dataset is created using a combination of web scraping and ChatGPT-generated sentences containing mountain names. The final dataset is stored in `data/mountain_sentences.csv`.

## Customization

This project can be extended to recognize other types of named entities by generating a corresponding dataset using OpenAI's model and adjusting the `prepare_data` function in `train_model.py`.

## Requirements

See `requirements.txt` for a list of dependencies.

## License

This project is licensed under the MIT License.
