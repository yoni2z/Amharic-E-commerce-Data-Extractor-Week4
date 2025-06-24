import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_MODE'] = 'disabled'
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report
import numpy as np
import torch
import logging
import argparse
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Train NER model')
parser.add_argument('--model_name', type=str, required=True, help='Model name')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
args = parser.parse_args()

def load_conll(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    token, label = line.strip().split()
                    current_sentence.append(token)
                    current_labels.append(label)
                else:
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence, current_labels = [], []
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
        return Dataset.from_dict({'tokens': sentences, 'ner_tags': labels})
    except Exception as e:
        logging.error(f"Error loading CoNLL: {e}")
        raise

logging.info("Loading CoNLL dataset")
dataset = load_conll('/content/labeled_data.conll')

label_list = sorted(set(label for sent in dataset['ner_tags'] for label in sent))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

def convert_labels_to_ids(example):
    example['ner_tags'] = [label2id[label] for label in example['ner_tags']]
    return example

dataset = dataset.map(convert_labels_to_ids)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_val_test = dataset.train_test_split(train_size=train_size, test_size=val_size+test_size, seed=42)
val_test = train_val_test['test'].train_test_split(train_size=val_size/(val_size+test_size), seed=42)
dataset_dict = DatasetDict({
    'train': train_val_test['train'],
    'validation': val_test['train'],
    'test': val_test['test']
})

logging.info(f"Loading tokenizer for {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

logging.info("Tokenizing dataset")
tokenized_dataset = dataset_dict.map(tokenize_and_align_labels, batched=True)

logging.info(f"Saving tokenized dataset for {args.model_name}")
try:
    os.makedirs(f'/content/data/processed/tokenized_dataset_{args.model_name.split("/")[-1]}', exist_ok=True)
    tokenized_dataset.save_to_disk(f'/content/data/processed/tokenized_dataset_{args.model_name.split("/")[-1]}')
except Exception as e:
    logging.error(f"Error saving tokenized dataset: {e}")
    raise

logging.info(f"Loading model {args.model_name}")
model = AutoModelForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    return {
        'precision': results['weighted avg']['precision'],
        'recall': results['weighted avg']['recall'],
        'f1': results['weighted avg']['f1-score']
    }

def measure_inference_time(model, tokenizer, text="Adidas SAMBAROSE ዋጋ 3300 ብር መገናኛ"):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs).logits
    return time.time() - start_time

training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='/content/logs',
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics
)

logging.info(f"Starting training for {args.model_name}")
trainer.train()

logging.info(f"Evaluating {args.model_name}")
eval_results = trainer.evaluate()
print(f"Evaluation Results for {args.model_name}:")
print(eval_results)

inference_time = measure_inference_time(model, tokenizer)
print(f"Inference Time for {args.model_name}: {inference_time:.4f} seconds")

logging.info(f"Saving model to {args.output_dir}")
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)  # Ensure tokenizer is saved
logging.info(f"Model and tokenizer saved to {args.output_dir}")

logging.info(f"Generating test set predictions for {args.model_name}")
predictions = trainer.predict(tokenized_dataset['test'])
print(f"Test Set Predictions for {args.model_name}:")
print(predictions.metrics)