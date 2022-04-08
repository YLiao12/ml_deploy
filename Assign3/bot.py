# Yongqing LIAO 1155161159

import time
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.dispatcher import run_async
from telegram.ext.filters import Filters
import random

import base64
import json
import socket
from io import BytesIO

from queue import Queue
from threading import Thread
from PIL import Image


from joblib import load


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

def handlePhoto(update: Update, context: CallbackContext):
    # bot.download_file(msg['photo'][-1]['file_id'], 'file.png')
    # Open the image.
    file = context.bot.getFile(update.message.photo[-1].file_id)
    # i = update.message.photo[-1]
    chat_id = update.message.chat_id
    file.download('photo.png')
    # Put to the queue.
    i = Image.open('photo.png')

    update.message.reply_text(chat_id)

    message_to_predict = {'image': i, 'chat_id': chat_id}
    image_queue.put(message_to_predict)
    

def send_to_predict(image_queue, output_queue):
    """Send images in the input queue to the server for prediction.
    Then receive the result, and put it to output queue.
    :param image_queue: Queue for incoming images (with chat ID).
    :param output_queue: Queue for sending prediction result back.
    """
    # Waiting for incoming images.
    while True:
        if not image_queue.empty():
            # Predict all images in the queue.
            # Get image from queue.
            print("image sent")
            incoming_message = image_queue.get()
            # TCP socket initialize.
            soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            soc.settimeout(5)
            soc.connect(('localhost', 8888))
            # Encode the image in base64.
            buffered = BytesIO()
            image = incoming_message['image']
            image.save(buffered, format='PNG')
            encoded_image = base64.b64encode(buffered.getvalue())
            data_send = json.dumps(dict({'image': encoded_image.decode('ascii'), 'chat_id': incoming_message['chat_id']}))
            # TCP client: encoded image send to the server. Waiting for receiving predictions.
            updater.dispatcher.bot.send_message(chat_id = incoming_message['chat_id'], text="image get!")
            terminate = '##END##'
            data_send += terminate
            soc.sendall(str.encode(data_send, 'utf8'))
            chunks = []
            while True:
                current_data = soc.recv(8192).decode('utf8', 'strict')
                if terminate in current_data:
                    chunks.append(current_data[:current_data.find(terminate)])
                    break
                chunks.append(current_data)
                if len(chunks) > 1:
                    last_pair = chunks[-2] + chunks[-1]
                    if terminate in last_pair:
                        chunks[-2] = last_pair[:last_pair.find(terminate)]
                        chunks.pop()
                        break
            received_data = ''.join(chunks)
            # JSON decode.
            decoded_data = json.loads(received_data)
            # Format
            predictions = ''
            idx = 1
            for item in decoded_data['predictions']:
                predictions += 'The bird is {}. \n'.format(item['label'])
                idx += 1
            send_back = {
                'predictions': predictions,
                'chat_id': incoming_message['chat_id']
            }
            # Put to queue.
            output_queue.put(send_back)

def send_predictions_back(output_queue):
    """Keep polling the output queue, send back the predictions to users.
    :param output_queue: Queue variable.
    """
    # Waiting for incoming predictions.
    while True:
        if not output_queue.empty():
            # Send all predictions back.
            send_back = output_queue.get()
            # Send message.
            updater.dispatcher.bot.send_message(chat_id = send_back['chat_id'], text=send_back['predictions'])

if __name__ == "__main__":
    
    # Provide your bot's token
    # , request_kwargs={'proxy_url': 'socks5://127.0.0.1:1080/'}
    updater = Updater("5192942243:AAFWOLvMIV_QLjRMmERIC8LDh2PJ_O1Lt3k", use_context=True, request_kwargs={'proxy_url': 'socks5://127.0.0.1:1080/'})

    # In assignment, if you need to load the model, load it here
    image_queue = Queue()
    output_queue = Queue()

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_handler(CommandHandler('hello', hello))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, handlePhoto))
    updater.start_polling()

    # Start threads.
    send_to_predict_thread = Thread(target=send_to_predict, args=(image_queue, output_queue), daemon=True)
    send_back_thread = Thread(target=send_predictions_back, args=(output_queue, ), daemon=True)
    send_to_predict_thread.start()
    send_back_thread.start()
    send_to_predict_thread.join()
    send_back_thread.join()

    