import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['WANDB_MODE'] = 'disabled'  # Disable W&B
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report
import numpy as np
import torch
import logging
import transformers

# Check transformers version
logging.info(f"Transformers version: {transformers.__version__}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load CoNLL file
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
            if current_sentence:  # Handle last sentence
                sentences.append(current_sentence)
                labels.append(current_labels)
        return Dataset.from_dict({'tokens': sentences, 'ner_tags': labels})
    except Exception as e:
        logging.error(f"Error loading CoNLL file: {str(e)}")
        raise

# Load dataset
logging.info("Loading CoNLL dataset")
dataset = load_conll('labeled_data.conll')

# Create label mappings
label_list = sorted(set(label for sent in dataset['ner_tags'] for label in sent))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Convert labels to IDs
def convert_labels_to_ids(example):
    example['ner_tags'] = [label2id[label] for label in example['ner_tags']]
    return example

dataset = dataset.map(convert_labels_to_ids)

# Split dataset
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

# Load tokenizer
logging.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        logging.info(f"Tokens: {tokenized_inputs['input_ids'][i]}")
        logging.info(f"Word IDs: {word_ids}")
        logging.info(f"Aligned Labels: {aligned_labels}")
        labels.append(aligned_labels)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

logging.info("Tokenizing dataset")
tokenized_dataset = dataset_dict.map(tokenize_and_align_labels, batched=True)

# Save tokenized dataset for Task 4
logging.info("Saving tokenized dataset")
try:
    tokenized_dataset.save_to_disk('data/processed/tokenized_dataset')
except Exception as e:
    logging.error(f"Error saving tokenized dataset: {str(e)}")

# Load model
logging.info("Loading model")
model = AutoModelForTokenClassification.from_pretrained(
    'xlm-roberta-base', 
    num_labels=len(label_list), 
    id2label=id2label, 
    label2id=label2id
)

# Define compute_metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    weighted_f1 = results['weighted avg']['f1-score']
    return {
        'precision': results['weighted avg']['precision'],
        'recall': results['weighted avg']['recall'],
        'f1': weighted_f1
    }

# Set training arguments
training_args = TrainingArguments(
    output_dir='results/fine_tuned_ner_model',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced for small dataset
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # Increased for better learning
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics
)

# Train model
logging.info("Starting training")
try:
    trainer.train()
except Exception as e:
    logging.error(f"Error during training: {str(e)}")
    raise

# Evaluate model
logging.info("Evaluating model")
eval_results = trainer.evaluate()
print("Evaluation Results:")
print(eval_results)

# Save model
logging.info("Saving model")
try:
    trainer.save_model('results/fine_tuned_ner_model')
except Exception as e:
    logging.error(f"Error saving model: {str(e)}")

# Save test dataset predictions
logging.info("Generating test set predictions")
predictions = trainer.predict(tokenized_dataset['test'])
print("Test Set Predictions:")
print(predictions.metrics)