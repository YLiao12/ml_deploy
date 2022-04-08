import time
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import random

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, Welcome to demo bot.")

def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Available Commands :-
    /hello - To reply you a hello
    Otherwise - To pick a random number between 0 and 1""")

def hello(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, Welcome to demo bot.")

def general(update: Update, context: CallbackContext):
    a = random.randint(0,10)
    reply = ""
    if a < 5:
        reply = "You obtain a small ({:.1f})".format(a/10.0)
    elif a > 5:
        reply = "You obtain a big ({:.1f})".format(a/10.0)
    else:
        reply = "You obtain the middle man"
    update.message.reply_text(reply)

if __name__ == "__main__":
    
    # Provide your bot's token
    updater = Updater("Your API token from botfather", use_context=True)

    # In assignment, if you need to load the model, load it here

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_handler(CommandHandler('hello', hello))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, general))
    updater.start_polling()