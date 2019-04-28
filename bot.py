from telegram.ext import Updater, MessageHandler, CommandHandler, Filters
from ranker import Ranker
import os.path
from query_processor import QueryProcessor
from intent_classifier import IntentClassifier
from secret import BOT_TOKEN
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def start_bot(path):
    ic_path = os.path.join(path, 'intent_classifier.pkl')
    intent_classifier = IntentClassifier(ic_path)
    ranker = Ranker()
    we_path = os.path.join(path, 'embeddings.emb.tsv')
    pe_path = os.path.join(path, 'post_embeddings.pkl4')
    idr_path = os.path.join(path, 'post_ids.pkl4')
    ranker.load(
        we_path=we_path,
        pe_path=pe_path,
        idr_path=idr_path
    )

    chatter = ChatBot("Stackoverflow IR")

    # Create a new trainer for the chatbot
    trainer = ChatterBotCorpusTrainer(chatter)
    # Train the chatbot based on the english corpus
    trainer.train("chatterbot.corpus.english")

    query_processor = QueryProcessor(intent_classifier, ranker, chatter)

    def handle_query(bot, update):
        print('Generating response')
        query = update['message']['text']
        response = query_processor.generate_response(query)
        bot.send_message(chat_id=update.message.chat_id, text=response)

    updater = Updater(token=BOT_TOKEN)
    query_handler = MessageHandler(Filters.text, handle_query)
    updater.dispatcher.add_handler(query_handler)
    print('Starting polling')
    updater.start_polling()
    updater.idle()