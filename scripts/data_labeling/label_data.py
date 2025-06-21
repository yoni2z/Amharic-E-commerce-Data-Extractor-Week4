import pandas as pd
import re
from transformers import AutoTokenizer
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Load data
data = pd.read_csv('data/raw/telegram_data.csv')
messages = data['Message'].dropna().sample(n=50, random_state=42).tolist()

# Expanded rule-based labeling
labeled_data = []
location_keywords = ['አዲስ', 'ቦሌ', 'ፒያሳ', 'መገናኛ', 'ሜክሲኮ', 'ጉርድ', 'ሾላ', 'ሆሊ', 'ሲቲ', 'ሴንተር', 'ድሬዳዋ', 'ገርጂ', '4ኪሎ', '5ኪሎ', '6ኪሎ', '🏢አድራሻ-ሜክሲኮ']
location_i_keywords = ['አበባ', '3ኛ', 'ፎቅ', 'አሸዋ', 'ሚና', 'ኮሜርስ', 'ጀርባ', 'መዚድ', 'ፕላዛ', 'ኢምፔሪያል', 'ከሳሚ', 'ህንፃ', 'ጎን', 'አልፎዝ', 'ፕላዛ', 'ግራውንድ', 'ቅድስት', 'ስላሴ', 'ህንፃ']
product_keywords = ['ጫማ', 'ልብስ', 'ስልክ', 'ቦርሳ', 'ቲሸርት', 'ጃኬት', 'መተኮሻ', 'ማበጠሪያ', 'ፔስትራ', 'መኪኊና', 'BMW', 'በፌራሪና', 'በፖርሽ', 'መኪና', 'ስጦታ', 'ስልክ', 'ስዕል', 'ደብተር']
price_keywords = ['ብር', 'ሺ', 'መቶ']
price_context = ['ዋጋ', 'ብር']
phone_context = ['ይደውሉ', 'ቁጥር']
product_context = ['በ', 'የ', '#ስጦታ']
stop_words = ['ይምጡ', 'ይደውሉ', 'http', 'T.me', 'tiktok', 'facebook', '@', '#']

for message in messages:
    tokens = message.split()  # Whitespace tokenization
    labels = []
    in_location = False  # Track multi-word locations
    location_count = 0  # Limit location length
    for i, token in enumerate(tokens):
        # Stop location tagging if too long or stop word encountered
        if in_location and (location_count > 5 or token in stop_words):
            in_location = False
            location_count = 0
        # Handle addresses
        if token == 'አድራሻ፦':
            labels.append('O')
            in_location = True
            location_count = 0
            logging.info(f"Started location at token: {token}")
            continue
        elif in_location and token not in ['.', ',', '1.', '2.']:
            labels.append('B-LOC' if not labels or labels[-1] not in ['B-LOC', 'I-LOC'] else 'I-LOC')
            location_count += 1
            logging.info(f"Labeled {token} as {'B-LOC' if not labels or labels[-1] not in ['B-LOC', 'I-LOC'] else 'I-LOC'}")
            continue
        # Handle phone numbers
        elif re.match(r'^\d+$', token) and len(token) >= 9 and (i > 0 and tokens[i-1] in phone_context):
            labels.append('O')
            in_location = False
            logging.info(f"Labeled {token} as O (phone number)")
            continue
        # Handle prices
        elif any(kw in token for kw in price_keywords) or (re.match(r'^\d+$', token) and len(token) <= 6 and
            (i > 0 and tokens[i-1] in price_context or (i < len(tokens)-1 and tokens[i+1] in price_context))):
            labels.append('B-PRICE' if not labels or labels[-1] not in ['B-PRICE', 'I-PRICE'] else 'I-PRICE')
            in_location = False
            logging.info(f"Labeled {token} as {'B-PRICE' if not labels or labels[-1] not in ['B-PRICE', 'I-PRICE'] else 'I-PRICE'}")
            continue
        # Handle products
        elif token in product_keywords or (token.startswith('#') and any(kw in token for kw in ['BMW', 'በፌራሪና', 'በፖርሽ', 'መኪና'])):
            labels.append('B-Product')
            in_location = False
            logging.info(f"Labeled {token} as B-Product")
        elif i > 0 and labels[i-1] in ['B-Product', 'I-Product'] and (token.startswith('የ') or token in product_context):
            labels.append('I-Product')
            in_location = False
            logging.info(f"Labeled {token} as I-Product")
        # Handle locations
        elif token in location_keywords:
            labels.append('B-LOC')
            in_location = False
            logging.info(f"Labeled {token} as B-LOC")
        elif token in location_i_keywords and i > 0 and labels[i-1] in ['B-LOC', 'I-LOC']:
            labels.append('I-LOC')
            in_location = False
            logging.info(f"Labeled {token} as I-LOC")
        else:
            labels.append('O')
            in_location = False
            logging.info(f"Labeled {token} as O")
    labeled_data.append(list(zip(tokens, labels)))

# Save to CoNLL format
with open('data/processed/labeled_data.conll', 'w', encoding='utf-8') as f:
    for sentence in labeled_data:
        for token, label in sentence:
            f.write(f"{token} {label}\n")
        f.write("\n")

# Save to CSV for manual review
review_data = []
for i, sentence in enumerate(labeled_data):
    for token, label in sentence:
        review_data.append({'Sentence': i, 'Token': token, 'Label': label})
pd.DataFrame(review_data).to_csv('data/processed/labeled_data_review.csv', encoding='utf-8', index=False)

# Print sample for validation
print("Sample Labeled Messages:")
for sentence in labeled_data[:3]:
    for token, label in sentence:
        print(f"{token} {label}")
    print()