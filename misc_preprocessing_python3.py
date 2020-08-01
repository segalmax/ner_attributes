import itertools
import nltk
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer as twt
from sklearn.model_selection import train_test_split
import io


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# todo: add func to return number of distinct tags
class SentenceGetter(object):
    def __init__(self, data):
        self.data = data

    def prepare_input(self):
        import ast
        output = []
        for ix, row in self.data.iterrows():
            doc = row['product_name']
            att_dict = ast.literal_eval(row['attrs_indexes_dict'])
            sent_words_pos_labels = get_words_pos_labels(doc, att_dict)
            sent_words_pos_iob_labels = get_iob_labels(sent_words_pos_labels)
            output.append(sent_words_pos_iob_labels)
        return output


def get_words_pos_labels(doc, att_dict):
    position_to_word = {v: k for k, v in att_dict.items()}
    words = nltk.word_tokenize(doc)
    pos_tags = nltk.pos_tag(words)
    token_indexes = list(twt().span_tokenize(doc))
    words_pos_labels = []
    for (word, part_of_speech), index_in_doc in zip(pos_tags, token_indexes):
        label = position_to_word.get(index_in_doc, 'O')
        words_pos_labels.append((word, part_of_speech, label))
    return words_pos_labels

def get_iob_labels(sent_words_pos_labels):
    list_cycle = itertools.cycle(sent_words_pos_labels)
    next(list_cycle)
    for i, (_ , _ ,label) in enumerate(sent_words_pos_labels[:-1]):
        sent_words_pos_labels[i] = list(sent_words_pos_labels[i])
        next_element = next(list_cycle)
        if label == 'O':
            continue
        elif label != 'O' and next_element[2] == label:
            sent_words_pos_labels[i][2] = 'B-{}'.format(label)
            while next(list_cycle)[2] == label:
                sent_words_pos_labels[i+1][2] = 'I-{}'.format(label)
                i+=1
        elif label != 'O' and next_element[2] == 'O':
            sent_words_pos_labels[i][2] = 'B-{}'.format(label)
        sent_words_pos_labels[i] = tuple(sent_words_pos_labels[i])
    return sent_words_pos_labels


import numpy as np
import nltk
nltk.download('stopwords')


# nltk.download('conll2002') # dataset


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


# employ crf model
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def crf_pipeline(X_train, y_train, X_test, y_test):
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True)
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))


def make_line_separated_format(sentences):
    with io.open("lstm_format.txt", "x", encoding="utf-8") as f:
        for sentence in sentences:
            # f.writelines(("%s    %s\n" % str(word, label) for word, pos, label in sentence))
            f.writelines([(word + "\t" + label + "\n") for word, pos, label in sentence])
            f.write("\n")


if __name__ == '__main__':
    # main_create_inedexed()
    data = pd.read_csv(r"data\computer_with_indexes_200720-183555.csv", encoding="utf-8")
    # data = data.fillna(method="ffill")

    print("Number of sentences: ", len(data))

    all_tokens = [nltk.word_tokenize(s) for s in data["product_name"].values]
    distinct_words = set(list(itertools.chain.from_iterable(all_tokens)))
    n_words = len(distinct_words)
    print("Number of words in the dataset: ", n_words)
    getter = SentenceGetter(data)
    # Get all the sentences
    sentences = getter.prepare_input()
    # convert sentences variable to tokens separated by new line and sentences
    make_line_separated_format(sentences)
    # divide to test and train sets
    test_index = len(sentences)//4
    test_set = sentences[:test_index]
    train_set = sentences[test_index:]

    X_train = [sent2features(s) for s in train_set]
    y_train = [sent2labels(s) for s in train_set]

    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]
    crf_pipeline(X_train, y_train, X_test, y_test)
