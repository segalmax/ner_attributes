# -*- coding: utf-8 -*-
"""BiLSTM_NER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1chQoyQ3eose58ZIpuf-Bu6wWwO_JepwE
"""

import pandas as pd
import numpy as np
from collections import Counter
import itertools
import nltk
nltk.download('punkt')
from nltk.tokenize import TreebankWordTokenizer as twt
import seaborn as sns
import matplotlib.pyplot as plt

def get_product_tokens_and_tags(product_doc, att_dict):
  try:
    tokens, tags = [], []
    position_to_word = {v: k for k, v in att_dict.items()}
    words = nltk.word_tokenize(product_doc)
    token_indexes = list(twt().span_tokenize(product_doc))
    for word, index_in_doc in zip(words, token_indexes):
        tokens.append(word)
        tag = position_to_word.get(index_in_doc, 'O')
        tags.append(tag)
    return tokens, tags
  except ValueError as e:
    print(e)  


def tag_presence(data_for_presence, find_tag):
    for token, tag in data_for_presence:
        if tag == find_tag:
            return True
    return False


class BiLSTMClassifier(object):
    def __init__(self, df):
        self.df = df
        self.tokens, self.tags = self.prepare_input()

    def prepare_input(self):
        import ast
        all_tokens = []
        all_tags = []
        # self.df = self.df[self.df['attrs_indexes_dict'] != u'{}']
        for _, row in self.df.iterrows():
            product_doc = row['product_name']
            att_dict = ast.literal_eval(row['attrs_indexes_dict'])
            try:
                tokens, tags = get_product_tokens_and_tags(product_doc, att_dict)
            except:
                continue    
            all_tokens.append(tokens)
            all_tags.append(tags)
        return all_tokens, all_tags

    # todo: can't import seaborn/matplotlib, apt-get install python-tk did not work; need to solve this for
    #  visualisations
    def sentence_length_distribution_analysis(self):
        lengths = list(map(lambda x: len(x), self.df['product_name']))
        ax = sns.distplot(lengths)
        ax.set(xlabel="Number of tokens in a sentences", ylabel="% of sentences")
        print ("Median: ", np.median(lengths))
        print ("Average: ", round(np.mean(lengths), 2))

    # can be used on self.tags or self.tokens
    def summary(self, item_list):
        flat_list = list(itertools.chain.from_iterable(item_list))
        cnt = Counter()
        for word in flat_list:
            cnt[word] += 1
        count_dict = dict(cnt)
        count_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        print("Number of unique items: ", len(count_items))
        print("Average count: ", round(len(item_list) / len(count_items)), "\n")
        total_items = len(item_list)
        proportion_list = []
        xlabels = []
        for i, (key, value) in enumerate(count_items):
            proportion = round(value * 100 / total_items, 2)
            proportion_list.append(proportion)
            xlabels.append(key)
            # print(key, " ---> ", proportion, "%")
        sns.set(style="whitegrid")
        chart = sns.barplot(xlabels, proportion_list, orient="v")
        plt.xlabel("Tokens/Tag")
        plt.ylabel("Percentage of total Tokens/Tags")
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

    # might be redundant since we know all words lengths are max 2-3 words
    def average_entity_length(self):
        tokens_tags = zip(self.tags, self.tokens)
        entities_lengths = []
        for tags, tokens in tokens_tags:
            for ix, tag in enumerate(tags):
                tag_dict = {tag: len(tokens[ix])}
                entities_lengths.append(tag_dict)
        flat_tags_list = list(itertools.chain.from_iterable(self.tags))
        cnt = Counter()
        for word in flat_tags_list:
            cnt[word] += 1
        count_dict = dict(cnt)
        for tag in set(flat_tags_list):
            total_length = 0
        print ('test')

    def tag_presence_percent(self):
        tags_counter = dict(Counter(list(itertools.chain.from_iterable(self.tags))))
        tags_counter = sorted(tags_counter.items(), key=lambda x: x[1], reverse=True)
        for label, total_count in tags_counter:
            num_of_sentences_with_label = len(list(filter(lambda x: tag_presence(x, label), self.tags, self.tokens)))
            print("Percentage of sentences having " + label[2:], " are: ", round(num_of_sentences_with_label*100/len(self.df),2), "%")

    def words_count(self, limit):
        tokens_dict = dict(Counter(self.tokens))
        tokens_count = tokens_dict.items()
        prop_list = []
        print("Vocabulary Size: ", len(tokens_count))
        for i in range(limit):
            tokens_filtered = len(list(filter(lambda x: x[1]<=i, tokens_count)))
            prop_list.append(round(tokens_filtered*100/len(tokens_count),2))
        plt.plot(prop_list)
        plt.xlabel("Counts")
        plt.ylabel("Proportion of Vocabulary (%)")

### MAIN CELL ###
import os
# path = '/content/drive/My Drive/Colab/Data sets/'
fn_indexed = 'data\computer_with_indexes_200720-183555.csv'
df_indexed = pd.read_csv(fn_indexed, encoding='utf-8')
classifier = BiLSTMClassifier(df_indexed)
tokens, tags = classifier.prepare_input()
classifier.sentence_length_distribution_analysis()
classifier.summary(classifier.tags)
classifier.tag_presence_percent()

# ### DEBUG ###
# # classifier.tag_presence_percent()
# tags_counter = dict(Counter(list(itertools.chain.from_iterable(classifier.tags))))
# tags_counter = sorted(tags_counter.items(), key=lambda x: x[1], reverse=True)
# all_data = zip(classifier.tags, classifier.tokens)
# def get_tokens_tags():
#     for tag, token in all_data:
#         yield token, tag
# data_generator = get_tokens_tags()
# for key, value in tags_counter:
#     sentence_with_tag = len(list(filter(lambda x: tag_presence(x, key), data_generator)))
#     print("Percentage of sentences having " + key[2:], " are: ", round(sentence_with_tag*100/len(classifier.df),2), "%")
# for token, tag in zip(classifier.tags, classifier.tokens):
#   print(token, tag)

# def tag_presence(data, find_tag):
#     for token, tag in data:
#         if tag == find_tag:
#             return True
#     return False
# for i, j in data_generator():
#     print(j,i)