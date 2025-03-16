from telethon import TelegramClient
import asyncio
import json
import os

api_id = int(os.getenv('API_ID'))
api_hash = os.getenv('API_HASH')
client = TelegramClient('session_name', api_id, api_hash)

channels = ['Раньше всех. Ну почти.', 'РИА Новости', 'Varlamov News', 'Медуза — LIVE']
limit = 5

async def fetch_messages(channel, limit):
    """ Fetch messages from single channel """
    try:
        message_list = []
        async for message in client.iter_messages(channel, limit):
            message_list.append(message.text)
        return channel, message_list
    except Exception as e:
        print(f'Error fetching messages from {channel}: {e}') 

async def main():
    """Fetching messages from all channels"""
    try: 
        dialogs = await client.get_dialogs()
        tasks = [fetch_messages(channel, limit) for channel in channels]
        results = await asyncio.gather(*tasks)

        post_history = {}
        for channel, messages in results: 
            if messages: 
                post_history[channel] = messages
                print(post_history[channel])
        with open('./post_history.json', 'w') as f:
            json.dump(post_history, f, ensure_ascii=False)
    except Exception as e: 
        print(f'Error: {e}')


if __name__=='__main__':
    client.start()
    client.loop.run_until_complete(main())
    client.disconnect()
