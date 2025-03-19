from telethon import TelegramClient
import asyncio
import json
import os

api_id = int(os.getenv('API_ID'))
api_hash = os.getenv('API_HASH')
client = TelegramClient('session_name', api_id, api_hash)

async def a():
    dialogs = await client.get_dialogs()
    b = await client.get_entity('РИА Новости')
    print(b)

client.start()
client.loop.run_until_complete(a())