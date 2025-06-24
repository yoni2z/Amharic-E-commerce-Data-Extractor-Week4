import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_MODE'] = 'disabled'
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from seqeval.metrics import classification_report
import torch
import time
import pandas as pd
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def measure_inference_time(model, tokenizer, text="Adidas SAMBAROSE ዋጋ 3300 ብር መገናኛ", num_runs=100):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs).logits
        total_time += time.time() - start_time
    return total_time / num_runs

def evaluate_model(model, tokenizer, dataset, id2label):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    true_labels, pred_labels = [], []
    for example in dataset:
        inputs = tokenizer(example['tokens'], is_split_into_words=True, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0]
        labels = example['labels']
        word_ids = inputs.word_ids()
        example_true, example_pred = [], []
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if word_ids[i] is not None and label != -100:
                example_true.append(id2label[label])
                example_pred.append(id2label[pred.item()])
        true_labels.append(example_true)
        pred_labels.append(example_pred)
    results = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    return {
        'precision': results['weighted avg']['precision'],
        'recall': results['weighted avg']['recall'],
        'f1': results['weighted avg']['f1-score']
    }

def get_model_size(model_dir):
    total_size = 0
    for file in glob.glob(f"{model_dir}/*"):
        total_size += os.path.getsize(file)
    return total_size / (1024 ** 2)

logging.info("Loading tokenized dataset")
tokenized_dataset = load_from_disk('file:///content/data/processed/tokenized_dataset_xlm-roberta-base')

# Define models and their paths
models = [
    {'name': 'XLM-RoBERTa', 'path': '/content/results/fine_tuned_ner_model', 'model_name': 'xlm-roberta-base'},
    {'name': 'mBERT', 'path': '/content/results/fine_tuned_mbert', 'model_name': 'bert-base-multilingual-cased'},
    {'name': 'DistilBERT', 'path': '/content/results/fine_tuned_distilbert', 'model_name': 'distilbert-base-multilingual-cased'}
]

comparison = []
for config in models:
    logging.info(f"Evaluating {config['name']}")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Attempt to load from the root directory
        model_path = config['path']
        if not os.path.exists(os.path.join(model_path, 'config.json')) or not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
            # Fallback to the latest checkpoint if root directory is incomplete
            checkpoints = glob.glob(os.path.join(model_path, 'checkpoint-*'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                model_path = latest_checkpoint
                logging.info(f"Fallback to latest checkpoint: {latest_checkpoint}")
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Evaluate model
        eval_results = evaluate_model(model, tokenizer, tokenized_dataset['validation'], model.config.id2label)
        inference_time = measure_inference_time(model, tokenizer)
        model_size = get_model_size(model_path)
        
        comparison.append({
            'Model': config['name'],
            'F1-Score': eval_results['f1'],
            'Precision': eval_results['precision'],
            'Recall': eval_results['recall'],
            'Inference Time (s)': inference_time,
            'Model Size (MB)': model_size
        })
    except Exception as e:
        logging.error(f"Failed to evaluate {config['name']}: {e}")

# Create and save comparison table
comparison_df = pd.DataFrame(comparison)
print("Model Comparison Table:")
print(comparison_df)
comparison_df.to_csv('/content/results/model_comparison.csv', index=False)
logging.info("Comparison table saved to /content/results/model_comparison.csv")