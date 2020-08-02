import itertools
import nltk
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer as twt
from sklearn.model_selection import train_test_split
import io
import re
import ast


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# todo: add func to return number of distinct tags
class SentenceGetter(object):
    def __init__(self, data):
        self.data = data

    def prepare_input(self):
        output = []
        for ix, row in self.data.iterrows():
            doc = row['product_name']
            att_dict = ast.literal_eval(row['attrs_indexes_dict'])
            sent_words_pos_labels = get_words_pos_labels(doc, att_dict)
            sent_words_pos_iob_labels = get_iob_labels_updated(sent_words_pos_labels)
            output.append(sent_words_pos_iob_labels)
        return output


def get_label(position_to_entity, word_index_in_doc):
    word_start_in_doc, word_end_in_doc = word_index_in_doc
    label = position_to_entity.get(word_index_in_doc, 'O')
    if label != 'O':
        return label
    for (entity_start, entity_end), label_in_attr_dict in position_to_entity.items():
        if entity_start <= word_start_in_doc and entity_end>=word_end_in_doc:
            return label_in_attr_dict
    return label


def get_words_pos_labels(doc, att_dict):
    position_to_entity = {v: k for k, v in att_dict.items()}
    word_delimiter_pattern = re.compile(r"[^,.:;+/()' ]+")
    words = re.findall(word_delimiter_pattern, doc)
    pos_tags = nltk.pos_tag(words)
    token_indexes = [(i.start(), i.end()) for i in re.finditer(word_delimiter_pattern, doc)]
    words_pos_labels = []
    assert len(pos_tags)   == len(token_indexes)
    for (word, part_of_speech), word_index_in_doc in zip(pos_tags, token_indexes):
        label = get_label(position_to_entity, word_index_in_doc)
        words_pos_labels.append((word, part_of_speech, label))
    return words_pos_labels


# todo: check why I- labels are marked as I-B- and fix
def get_iob_labels_updated(sent_words_pos_labels):
    labels_list = [label if label != 'O' else 'O' for (_, _, label) in sent_words_pos_labels]
    grouped_list = [list(grp) for k, grp in itertools.groupby(labels_list)]
    for group in grouped_list:
        if None in group:
            continue
        if len(group) == 1:
            group[0] = 'B-{}'.format(group[0])
        else:
            # iob_tagged_group = ['B-{}'.format(group[0])] + ['I-{}'.format(group[i]) for i,_ in enumerate(group[1:])]
            # group = iob_tagged_group
            group = map(lambda x: 'B-{}'.format(x) if group.index(x) == 0 else 'I-{}'.format(x), group)
    iob_labels_list = list(itertools.chain.from_iterable(grouped_list))
    word_pos_list = [(word, pos) for (word, pos, _) in sent_words_pos_labels]
    sent_words_pos_iob_labels = [(word, pos, iob_label) for (word, pos), iob_label in
                                 zip(word_pos_list, iob_labels_list)]
    return sent_words_pos_iob_labels


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
    data = pd.read_csv(r"data/all_cms_indexed.csv", encoding="utf-8")
    # data = data.fillna(method="ffill")

    print("Number of sentences: ", len(data))

    all_tokens = [nltk.word_tokenize(s) for s in data["product_name"].values]
    distinct_words = set(list(itertools.chain.from_iterable(all_tokens)))
    n_words = len(distinct_words)
    print("Number of words in the dataset: ", n_words)
    ### test
    doc = 'HP Pavilion 20-c416il 19.5-inch Full HD All-in-One Desktop (Celeron J4005/4GB/1 TB/DOS/Wireless Keyboard & Mouse), Black'
    att_dict = ast.literal_eval(
        "{u'color': (115, 120), u'processor_name': (60, 67), u'system_memory': (74, 77), u'operating_system': (83, 86), u'features': (32, 39)}")
    sent_words_pos_labels = get_words_pos_labels(doc, att_dict)
    sent_words_pos_iob_labels = get_iob_labels_updated(sent_words_pos_labels)
    ###
    getter = SentenceGetter(data)
    # Get all the sentences
    sentences = getter.prepare_input()
    # convert sentences variable to tokens separated by new line and sentences
    make_line_separated_format(sentences)
    # divide to test and train sets
    test_index = len(sentences) // 4
    test_set = sentences[:test_index]
    train_set = sentences[test_index:]

    X_train = [sent2features(s) for s in train_set]
    y_train = [sent2labels(s) for s in train_set]

    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]
    crf_pipeline(X_train, y_train, X_test, y_test)
