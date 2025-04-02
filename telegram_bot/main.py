import os
from pytz import utc
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import sqlite3
from telegram_bot.helpers.database import get_user_channels, insert_user_channel, get_channel_status, change_status, get_channel_name_by_id, unsubscribe_by_id, get_all_users
from telegram_bot.helpers.modeling import create_newsletter
from apscheduler.schedulers.background import BackgroundScheduler
import time

BOT_TOKEN = os.getenv('BOT_TOKEN')
scheduler = BackgroundScheduler(timezone=utc)

## Handle forwarded messages
async def handle_forwarded_messages(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    message = update.message

    ## Check if message has forward origin
    if message.forward_origin: 
        origin = message.forward_origin
        if origin.type == "channel": 
            channel_id = origin.chat.id
            channel_name = origin.chat.title

            ## Fetch user channels
            user_channels = get_user_channels(message.from_user.id, 'all')
            user_channel_names = [channel[0] for channel in user_channels]

            if origin.chat.title not in user_channel_names: 
                insert_user_channel(message.from_user.id, origin.chat.id, origin.chat.title) 
                await update.message.reply_text(f"Channel {origin.chat.title} has been added.")

            elif origin.chat.title in user_channel_names and get_channel_status(message.from_user.id, origin.chat.id) == 1: 
                await update.message.reply_text("This channel ID is already added.")
                
            else:
                change_status(message.from_user.id, origin.chat.id)
                await update.message.reply_text(f"Channel {origin.chat.title} has been added.")
        else: 
            await update.message.reply_text(f"Please forward a message from a channel.")
    else: 
            await update.message.reply_text(f"Please forward a message from a channel.")


## Command to list all user's subscriptions
async def list_my_channels(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try: 
        channels = get_user_channels(user_id, 'active')
        if not channels: 
            await update.message.reply_text('You have no subscriptions')
        else: 
            await update.message.reply_text(f"You added the following channels:\n"
                                             + "\n".join([f"- {channel[0]}" for channel in channels]))
    except Exception as e: 
        await update.message.reply_text(f'Error fetching channels: {e}')

# # Command to unsubsucribe
async def unsubscribe_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    channels = get_user_channels(user_id, 'active')

    if not channels: 
        await update.message.reply_text("You have no subscriptions.")
        return
    
    try: 
    # Create buttons for each channel
        keyboard = [
            [InlineKeyboardButton(channel[0], callback_data=str(channel[1]))]
            for channel in channels
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "Select a channel to unsubscribe:",
            reply_markup=reply_markup
            )
    except Exception as e: 
        await update.message.reply_text(f'Error: {e}')

async def unsubscribe_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Extract channel_id from callback data
    channel_id = int(query.data)
    user_id = query.from_user.id
    
    ## Get channel name from db
    channel_name = get_channel_name_by_id(channel_id)
    if not channel_name:
        await query.edit_message_text("Invalid selection. Please try again.")
        return
    
    ## Unsubscribe from the selected channel
    unsubscribe_by_id(user_id, channel_id)

    ## Inform the user
    await query.edit_message_text(f"You have unsubscribed from the selected channel {channel_name}.")

async def send_newsletters():
    """Send newsletters to all users in the database"""
    ## Fetch all user IDs from the database
    users = get_all_users() 
    for user_id in users:
        ## Generate custom newsletter
        all_articles = await create_newsletter(user_id)
        if not all_articles: 
            await application.bot.send_message(chat_id=user_id, text="No important news!")
        for article in all_articles: 
            try:
                await application.bot.send_message(chat_id=user_id, text=article)
            except Exception as e:
                print(f"Failed to send newsletter to {user_id}: {e}")

# Schedule the job at 7 AM UTC
scheduler.add_job(send_newsletters, 'cron', hour=7, minute=0)
scheduler.start()

async def send_newsletters_command(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    await send_newsletters()
    await update.message.reply_text("Newsletters sent to all users!")

if __name__ == '__main__': 
    print('Polling...')
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.FORWARDED, handle_forwarded_messages))
    application.add_handler(CommandHandler("my_channels", list_my_channels))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe_menu))
    application.add_handler(CallbackQueryHandler(unsubscribe_handler))
    application.add_handler(CommandHandler("send_newsletters_admin", send_newsletters_command))
    application.run_polling()