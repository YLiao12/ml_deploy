import time
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import random

from joblib import load

tfidf_model = load('tfidf_model.pkl')

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, Welcome to demo bot.")

def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Available Commands :-
    /hello - To reply you a hello
    /asmt2model1 - To test fake or real news
    Otherwise - To pick a random number between 0 and 1""")

def hello(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, Welcome to demo bot create by lyq.")

def model1(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, what is the title of the news.")
    

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

def analyse(update: Update, context: CallbackContext):
    message = update.message
    text = message.text
    predict = tfidf_model.predict_proba([text])
    predict = predict.flatten().tolist()
    print(predict)

    if predict[1] < 0.4:
        predict_text = "fake"
        predict_precentage = round(predict[1], 2)
    elif predict[1] > 0.6:
        predict_text = "real"
        predict_precentage = round(predict[0], 2)
    else:
        predict_text = "not sure"
        predict_precentage = round(predict[0], 2)

    update.message.reply_text(predict_text)


if __name__ == "__main__":
    
    # Provide your bot's token
    updater = Updater("5192942243:AAFWOLvMIV_QLjRMmERIC8LDh2PJ_O1Lt3k", use_context=True, request_kwargs={'proxy_url': 'socks5://127.0.0.1:1080/'})

    # In assignment, if you need to load the model, load it here

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_handler(CommandHandler('hello', hello))
    updater.dispatcher.add_handler(CommandHandler('asmt2model1', model1))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, analyse))
    updater.start_polling()