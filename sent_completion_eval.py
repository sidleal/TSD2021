import pandas as pd
import itertools
from gensim.models import KeyedVectors


import argparse
import time
import csv
import re
import string
import pickle
from scipy import spatial
import numpy as np
import gensim

test_file = 'dataset_v1.tsv'

embeddings = {}

# embedding_file = 'bbp_fasttext_cbow_300d.txt'
# output_file = 'prediction_2_bbp_ft.csv'
# embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

# embedding_file = 'bbp_word2vec_cbow_300d.txt'
# output_file = 'prediction_2_bbp_w2v.csv'
# embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

# embedding_file = 'nilc_word2vec_cbow_s300.txt'
# output_file = 'prediction_2_nilc_w2v.csv'
# embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

# embedding_file = 'nilc_fasttext_cbow_s300.txt'
# output_file = 'prediction_2_nilc_ft.csv'
# embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

embedding_file = '../brwac_full_lsa_word_dict.pkl'
output_file = 'prediction_2_lsa.csv'
with open(embedding_file, 'rb') as f:
    embeddings = pickle.load(f)



def word2vec(tokens, embeddings):
    dim = embeddings['word'].size

    word_vec = []
    for word in tokens:
        if word.lower() in embeddings:
            word_vec.append(embeddings[word.lower()])
        else:
            word_vec.append(np.random.uniform(-0.001, 0.001, dim))
    return word_vec


def avg_similarity(vec, context_vec):
    print("------------------------------------->", len(vec))
    context_vec_avg = np.average(context_vec, axis=0)
    return 1 - spatial.distance.cosine(vec, context_vec_avg)


def calc_acc():
    map_answers = {}
    answers = 'dataset_v1_answers.tsv'
    with open(answers, 'r', encoding='utf-8') as f:
        dict_answers = csv.DictReader(f, dialect='excel-tab')
        for i, row in enumerate(dict_answers):
            map_answers['%s'%(i+1)] = row['answer']
            map_answers['%s_id'%(i+1)] = row['id']

    prediction = output_file
    hits = 0
    with open(prediction, 'r', encoding='utf-8') as f:
        dict_prediction = csv.DictReader(f)
        for i, row in enumerate(dict_prediction):
            sent_id = map_answers['%s_id'%row['id']]
            right_answer = map_answers[row['id']]
            predicted_answer = row['answer']
            print("--->", sent_id, predicted_answer, right_answer)
            if predicted_answer == right_answer:
                hits += 1

    print('Total acertos:', hits, " de 113 = ", hits * 100 / 113)


if __name__ == '__main__':


    start = time.time()
    print("Loading pretrained embeddings: {}".format(embedding_file))

    # Load pretrained word embeddings

    keys = ['a', 'b', 'c', 'd', 'e']
    choices = ['a', 'b', 'c', 'd', 'e']
    prediction = []

    print("Predicting answers")
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect='excel-tab')
        for i, row in enumerate(reader):
            question = row['question']
            translator = str.maketrans('', '', string.punctuation)
            question = question.translate(translator)
            tokens = question.split()

            print(tokens)

            # get word2vec embedding
            ques_vec = word2vec(tokens, embeddings)

            # calculate total word similarity
            scores = []
            candidates = [row[x] for x in keys]
            print(candidates)
            cand_vec = word2vec(candidates, embeddings)
            for word in cand_vec:
                s = avg_similarity(word, ques_vec)
                # print(s)
                scores.append(s)

            idx = scores.index(max(scores))
            ans = choices[idx]
            prediction.append(ans)

    with open(output_file, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id', 'answer'])
        for i, ans in enumerate(prediction):
            writer.writerow([str(i + 1), ans])
    print("Output prediction file: {}".format(output_file))

    print("Total run time: {}s".format(time.time() - start))

    calc_acc()