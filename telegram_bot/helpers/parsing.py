from telethon import TelegramClient
from telegram_bot.helpers.database import get_user_channels
from datetime import datetime, timedelta, timezone
import asyncio
import os

api_id = int(os.getenv('API_ID'))
api_hash = os.getenv('API_HASH')
client = TelegramClient('session_name', api_id, api_hash)

async def fetch_channel_names(user_id): 
    channels = get_user_channels(user_id, 'active')
    channel_names = [channel[0] for channel in channels]
    return channel_names

async def fetch_single_channel(channel, start):
    """ Fetch messages from single channel
      that were sent before 'start' timestamp."""
    try:
        message_list = []
        async for message in client.iter_messages(channel):
            if message.date < start:
                break
            else:
                message_list.append(message.text)
        print(f'Finished fetching messages from {channel}')
        return message_list
    except Exception as e:
        print(f'Error fetching messages from {channel}: {e}')
        return []

async def fetch_all_messages(user_id):
    """Fetching messages from all channels."""
    async with client:
        channels = await fetch_channel_names(user_id)

        ## Select cut-off timestamp
        start = datetime.now(timezone.utc) - timedelta(hours=24)

        ## Initialize posts list
        all_posts = []
        try:
            for channel in channels: 
                dialogs = await client.get_dialogs()
                messages = await fetch_single_channel(channel, start)
                if messages: 
                    all_posts.extend(messages)
            return all_posts
        except Exception as e: 
            print(f'Error: {e}')
            return []