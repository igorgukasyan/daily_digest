import pandas as pd
from telegram_bot.helpers.parsing import fetch_channel_names, fetch_single_channel, fetch_all_messages
from bert.preprocessing import remove_emojis
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import asyncio

async def get_posts(user_id):
    """Fetch all messages asynchronously"""
    messages = await fetch_all_messages(user_id)
    return messages

def clean_posts(posts):
    """Remove emojis from posts"""
    return [remove_emojis(post) for post in posts]

def calculate_embeddings(model, posts):
    """Calculate embeddings on cleaned posts"""
    cleaned_posts = clean_posts(posts)
    vectors = model.encode(cleaned_posts, batch_size=32)
    return pd.DataFrame(vectors)
    
async def create_newsletter(user_id):
    """Main async function"""
    with open('./bert/logit.pkl', 'rb') as f:
        logit = pickle.load(f)

    # Load sentence transformer model
    model = SentenceTransformer('sergeyzh/rubert-tiny-turbo')

    # Fetch messages asynchronously
    posts = await get_posts(user_id)

    # Calculate embeddings using the correct model
    embeddings = calculate_embeddings(model, posts)

    # Predict using loaded logit model
    res = logit.predict(embeddings)

    # 
    output_indices = np.where(res==1)[0]
    outputted_posts = [posts[i] for i in output_indices]
    
    return outputted_posts
