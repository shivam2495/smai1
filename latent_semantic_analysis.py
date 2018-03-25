import string
import numpy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os import listdir
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer

word_list = []
train_word_freq = []
train_file_names = []


def voted(x, y):
    d = len(x[0])
    m = len(x)
    n = 0
    w = [[[0 for _ in range(d)] for _ in range(len(listdir("train")))]]
    c = [1]
    for _ in tqdm(range(100)):
        for j in range(m):
            wtx = numpy.dot(numpy.array(w[n]), numpy.array(x[j]).T)
            max_ind = wtx.argmax()
            if y[j] != max_ind:
                temp_w = w[n]
                temp_w[y[j]] = [(x + y) for x, y in zip(temp_w[y[j]], x[j])]
                temp_w[max_ind] = [(x - y) for x, y in zip(temp_w[max_ind], x[j])]
                w.append(temp_w)
                c.append(1)
                n += 1
            else:
                c[n] += 1
    return w, c


for label in listdir("train"):
    for file in listdir("train/" + label):
        train_file_names.append(label + "_" + file)
        train_word_freq.append([])
for file_no in tqdm(range(len(train_file_names))):
    with open("train/" + train_file_names[file_no].split('_')[0] + "/" + train_file_names[file_no].split('_')[1],
              "rb") as f:
        for line in f:
            words = line.split()
            for word in words:
                raw_word = str(word, errors='ignore').translate(str.maketrans('', '', string.punctuation)).lower()
                if raw_word and raw_word not in set(stopwords.words('english')):
                    filtered_word = PorterStemmer().stem(raw_word)
                    if filtered_word not in word_list:
                        word_list.append(filtered_word)
                        for i in range(len(train_file_names)):
                            if i == file_no:
                                train_word_freq[i].append(1)
                            else:
                                train_word_freq[i].append(0)
                    else:
                        train_word_freq[file_no][word_list.index(filtered_word)] += 1
train_tfidf = TfidfTransformer().fit_transform(train_word_freq).toarray()
train_biased_tfidf = numpy.hstack((train_tfidf, numpy.ones((len(train_tfidf), 1))))
train_labels = [int(i.split('_')[0]) for i in train_file_names]
weights, votes = voted(train_biased_tfidf, train_labels)

# testing
test_file_names = []
test_word_freq = []
for label in listdir("test"):
    for file in listdir("test/" + label):
        test_file_names.append(label + "_" + file)
for file_no in tqdm(range(len(test_file_names))):
    test_word_freq.append([0 for _ in range(len(word_list))])
    with open("test/" + test_file_names[file_no].split('_')[0] + "/" + test_file_names[file_no].split('_')[1],
              "rb") as f:
        for line in f:
            words = line.split()
            for word in words:
                raw_word = str(word, errors='ignore').translate(str.maketrans('', '', string.punctuation)).lower()
                if raw_word and raw_word not in set(stopwords.words('english')):
                    filtered_word = PorterStemmer().stem(raw_word)
                    if filtered_word in word_list:
                        test_word_freq[file_no][word_list.index(filtered_word)] += 1
test_tfidf = TfidfTransformer().fit_transform(test_word_freq).toarray()
test_biased_tfidf = numpy.hstack((test_tfidf, numpy.ones((len(test_tfidf), 1))))
test_labels = [int(i.split('_')[0]) for i in test_file_names]
passed = 0
for row in tqdm(range(len(test_biased_tfidf))):
    votedsum = [0 for _ in range(len(listdir("test")))]
    for k in range(len(weights)):
        test_wtx = numpy.dot(numpy.array(weights[k]), numpy.array(test_biased_tfidf[row]).T)
        for i in range(len(votedsum)):
            votedsum[i] += votes[k] * test_wtx[i]
    if test_labels[row] == numpy.array(votedsum).argmax():
        passed += 1
print(passed / len(test_biased_tfidf))
