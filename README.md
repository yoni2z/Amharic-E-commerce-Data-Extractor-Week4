# Amharic-E-commerce-Data-Extractor-Week4
Project for 10 Academy Week 4: Building an Amharic NER system for EthioMart.

## Setup
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with Telegram API credentials.
4. Run scripts in `scripts/` for each task.

## Folder Structure
- `data/`: Raw and processed data.
- `scripts/`: Python scripts for tasks 1–6.
- `notebooks/`: Jupyter notebooks for exploration.
- `results/`: Model outputs and vendor scorecard.
- `docs/`: Submission PDFs.

## Tasks
1. Data Ingestion: `scripts/data_ingestion/telegram_scraper.py`
2. Data Labeling: `scripts/data_labeling/label_data.py`
3. Model Fine-Tuning: `scripts/model_training/fine_tune_ner.py`
4. Model Comparison: `scripts/model_comparison/compare_models.py`
5. Model Interpretability: `scripts/model_interpretability/interpret_model.py`
6. Vendor Scorecard: `scripts/vendor_scorecard/vendor_scorecard.py`