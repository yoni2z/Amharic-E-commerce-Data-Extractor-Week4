import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import unicodedata
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError, FloodWaitError
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# Initialize Telegram client
client = TelegramClient('scraping_session', api_id, api_hash)

# Function to preprocess Amharic text
def preprocess_amharic_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize Unicode characters (Ge'ez script)
    text = unicodedata.normalize('NFC', text)
    # Remove extra whitespace and special characters
    text = ' '.join(text.strip().split())
    return text

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        async for message in client.iter_messages(entity, limit=10000):
            media_path = None
            views = message.views if message.views else 0
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)
            
            # Preprocess message text
            processed_text = preprocess_amharic_text(message.message)
            
            # Write to CSV
            writer.writerow([
                channel_title,
                channel_username,
                message.id,
                processed_text,
                message.date.isoformat(),
                views,
                media_path
            ])
    except Exception as e:
        logging.error(f"Error scraping {channel_username}: {str(e)}")

async def main():
    try:
        await client.start(phone)
        logging.info("Telegram client started")
        
        # Create media directory
        media_dir = 'data/photos'
        os.makedirs(media_dir, exist_ok=True)
        
        # Load channels from Excel
        channels_df = pd.read_excel('data/raw/channels_to_crawl.xlsx')
        channels = channels_df['channels_to_crawl'].unique()[:8]  # Select 8 unique channels
        
        # Open CSV file
        with open('data/raw/telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Views', 'Media Path'])
            
            # Scrape each channel
            for channel in channels:
                logging.info(f"Scraping {channel}")
                await scrape_channel(client, channel, writer, media_dir)
                logging.info(f"Finished scraping {channel}")
                
    except SessionPasswordNeededError:
        logging.error("Two-factor authentication required. Please provide password.")
    except FloodWaitError as e:
        logging.error(f"Flood wait error: Please wait {e.seconds} seconds.")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())