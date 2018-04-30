from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import numpy as np
import pandas as pd
import re

# ##################################################################################### Generating vocab

corpus = pd.read_csv("abc_small.csv")
tokenizer = RegexpTokenizer(r'\w+')  # remove punctuation
stop_words = stopwords.words('english')
text_data = [line for line in corpus["text"] if line != '']
for line in tqdm(range(len(corpus))):
    line_token = tokenizer.tokenize(text_data[line].lower())
    clean_token = [re.sub(r'[^a-zA-Z]', '', word) for word in line_token]  # only alphabets
    stop_token = [word for word in clean_token if word not in stop_words if word != '']  # remove stop words
    stem_token = [str(PorterStemmer().stem(word)) for word in stop_token]  # stemmed
    text_data[line] = stem_token
vocab_total = set([words for sublist in text_data for words in sublist])
# print(vocab_total)
text_ID = []
for line in tqdm(range(len(text_data))):
    text_to_num_line = [list(vocab_total).index(words) for words in text_data[line]]
    text_ID.append(text_to_num_line)
# print(text_ID)

# ####################################################################################### Random init and other vectors

K = 20  # no of topics
V = len(vocab_total)  # no of words in vocabulary
D = len(text_ID)  # no of documents
word_topic_count = np.zeros((K, V))
doc_word_topic_assignd = [np.zeros(len(sublist)) for sublist in text_ID]  # dimension D * words in that doc
doc_topic_count = np.zeros((D, K))
for doc in tqdm(range(D)):
    for word in range(len(text_ID[doc])):
        doc_word_topic_assignd[doc][word] = np.random.choice(K, 1)
        word_topic = int(doc_word_topic_assignd[doc][word])
        word_doc_ID = text_ID[doc][word]
        word_topic_count[word_topic][word_doc_ID] += 1
for doc in tqdm(range(D)):
    for topic in range(K):
        topic_doc_vector = doc_word_topic_assignd[doc]
        doc_topic_count[doc][topic] = sum(topic_doc_vector == topic)

# ######################################################################################

alpha = 0.1
beta = 0.01
corpus_itter = 50
for itter in tqdm(range(corpus_itter)):
    for doc in range(D):
        for word in range(len(text_ID[doc])):
            init_topic_assign = int(doc_word_topic_assignd[doc][word])
            word_id = text_ID[doc][word]
            doc_topic_count[doc][init_topic_assign] -= 1
            word_topic_count[init_topic_assign][word_id] -= 1
            # ############### Gibb's sampling begin
            denom1 = sum(doc_topic_count[doc]) + K * alpha
            denom2 = np.sum(word_topic_count, axis=1) + V * beta
            numerator1 = [doc_topic_count[doc][col] for col in range(K)]
            numerator1 = np.array(numerator1) + alpha
            numerator2 = [word_topic_count[row][word_id] for row in range(K)]
            numerator2 = np.array(numerator2) + beta
            prob_topics = (numerator1 / denom1) * (numerator2 / denom2)
            prob_topics = prob_topics / sum(prob_topics)
            # ############### Gibb's sampling end
            update_topic_assign = np.random.choice(K, 1, list(prob_topics))
            doc_word_topic_assignd[doc][word] = update_topic_assign
            doc_topic_count[doc][init_topic_assign] += 1
            word_topic_count[init_topic_assign][word_id] += 1
theta = (doc_topic_count+alpha)
theta_row_sum = np.sum(theta, axis=1)
theta = theta/theta_row_sum.reshape((D, 1))  # probab of a doc falling in a given topic after running lda K*D
phi = (word_topic_count + beta)
phi_row_sum = np.sum(phi, axis=1)
phi = phi/phi_row_sum.reshape((K, 1))  # probab of a word falling in a given topic after running lda K*V
list_dict_topics = []
for topic in tqdm(range(K)):
    mydict = {}
    for word in range(V):
        mydict[list(vocab_total)[word]] = phi[topic][word]
    list_dict_topics.append(mydict)
doc_topic_final = []
for i in range(K):
    x = sorted([(value, key) for (key, value) in list_dict_topics[i].items()])[::-1][20:30]
    for j in x:
        print(j)
    print("")
