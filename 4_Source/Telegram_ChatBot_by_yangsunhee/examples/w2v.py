#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Simple Bot to reply to Telegram messages. This is built on the API wrapper, see
# echobot2.py to see the same example built on the telegram.ext bot framework.
# This program is dedicated to the public domain under the CC0 license.
import logging
import telegram
from telegram.error import NetworkError, Unauthorized
from time import sleep
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec

raw_corpus_fname = 'norm_2016-10-24_article_all.txt' # Fill your corpus file

class Word2VecCorpus:
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for line in f:
                line = line.strip().replace('\n', '')
                if not line:
                    continue
                yield line.split()

word2vec_corpus = Word2VecCorpus(raw_corpus_fname)
word2vec_model = Word2Vec(word2vec_corpus, size=150, min_count=10)


update_id = None

def main():
    global update_id
    # Telegram Bot Authorization Token
    bot = telegram.Bot('358456821:AAFkeA5G0dlZPNavaB16YSgD_VNndE4pkPA')

    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    try:
        update_id = bot.getUpdates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    while True:
        try:
            echo(bot)
        except NetworkError:
            sleep(1)
        except KeyError:
            sleep(1)
            # update.message.reply_text("sorry!!  not in vocabulary")
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1


def echo(bot):
    global update_id
    # Request updates after the last update_id
    for update in bot.getUpdates(offset=update_id, timeout=10):
        # chat_id is required to reply to any message
        chat_id = update.message.chat_id
        update_id = update.update_id + 1

        # update.message.reply_text("검색단어를 입력하세요? ex) 오바마,박근혜(코퍼스가 신문사설입니다.)")

        if update.message:  # your bot can receive updates without messages
            values = ',\n'.join(str(v) for v in word2vec_model.most_similar(update.message.text))
            
            # Reply to the message
            update.message.reply_text(values)
            update.message.reply_text("검색단어를 입력하세요? ex)오바마, 박근혜(코퍼스가 신문내용입니다.)")


if __name__ == '__main__':
    main()
