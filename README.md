#### Intro

This is full code I used to create a Telegram newsletter that only sends you important news.

#### How To Use

- Clone repo and cd into it
```python
git clone https://github.com/igorgukasyan/daily_digest
cd hack-interview
```
- Create a virtual environment and activate it
```
python3 -m venv .venv
source .venv/bin/activate
```
- Download requirements with 
```python
python3 -m pip install -r requirements.txt
```
- Create a local .db file
```python
import sqlite3
connection = sqlite3.connect("./telegram_bot/helpers/daily_digest.db")
cursor = connection.cursor()
## Create a table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        channel_id INTEGER NOT NULL,
        channel_name TEXT NOT NULL,
        status BOOLEAN NOT NULL DEFAULT TRUE
    )
""")
```
- Set environment variables. You only will need to set API_ID, API_HASH and BOT_TOKEN, all of which you can get from Telegram.
- Run main script
```python
python telegram_bot/main.py
```
- Done! Your bot is up and running.



