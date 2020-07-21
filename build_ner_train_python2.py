import itertools
import json
import random
from collections import Counter

import nltk
import pandas as pd
import spacy
from pathlib import Path
from spacy.lang.en import English
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from sklearn.model_selection import train_test_split

from comps.product_classifier.db_data_fetch import DataFetchFuncs
from lib.help_functions import df_to_csv, read_excel_range_to_df
from lib.time_utils import time_string
from itertools import izip, islice, tee


def consecutive_indexes(iterable, n):
    return izip(*(islice(it, i, None) for i, it in enumerate(tee(iterable, n))))


def tuples_have_overlapping_indexes(tuples_list):
    tuples_list = sorted(tuples_list, key=lambda x: x[0])
    indexes = list(consecutive_indexes(range(len(tuples_list)), 2))
    overlapping_tuples=[]
    for index_of_first, index_of_second in indexes:
        if not (tuples_list[index_of_first][1] < tuples_list[index_of_second][0]):
            overlapping_tuples.append((tuples_list[index_of_first],tuples_list[index_of_second]))
    return overlapping_tuples


def test_overlaps():
    test_ind1 = [(1, 3), (5, 7), (6, 9)]
    test_ind2 = [(1, 3), (5, 7), (8, 9)]
    test_ind3 = [(1, 5), (5, 7), (8, 9)]
    test_ind4 = [(5, 7), (1, 5), (8, 9)]
    test_ind5 = [(5, 7), (1, 3), (8, 9)]
    assert tuples_have_overlapping_indexes(test_ind1) == True
    assert tuples_have_overlapping_indexes(test_ind2) == False
    assert tuples_have_overlapping_indexes(test_ind3) == True
    assert tuples_have_overlapping_indexes(test_ind4) == True
    assert tuples_have_overlapping_indexes(test_ind5) == False


def dict_of_tuples_with_overlapping_indexes(d):
    list_of_indexes = list(itertools.chain(d.values()))
    return tuples_have_overlapping_indexes(list_of_indexes)


def map_cms_to_relevant_attribute_names():
    fn = 'tmp/cms_to_attributes.xlsx'
    df = read_excel_range_to_df(fn, 'Sheet1', 'a1', 'k17', True)
    d = dict()
    for cms_col in df.columns:
        d[trim(cms_col)] = [trim(x) for x in list(df[cms_col]) if not pd.isnull(x)]
    return d
    # d = {u'mini_pc': [u'processor_type', u'operating_system', u'ram', u'ram_type'],
    #      u'lens_cap': [u'lens_cap_type', u'filter_thread_size'],
    #      u'point_shoot_camera': [u'optical_zoom', u'digital_zoom', u'built_in_flash', u'video_resolution'],
    #      u'all_in_one': [u'processor_model', u'processor_brand', u'operating_system', u'graphic_memory', u'system_memory'],
    #      u'tablet': [u'connectivity', u'os', u'ram', u'sim_type'],
    #      u'lens_hood': [u'shape', u'filter_thread_size', u'mount_type'],
    #      u'camcorder': [u'sensor_type', u'optical_zoom', u'digital_zoom', u'video_quality'],
    #      u'lens': [u'lens_type', u'prime_zoom'],
    #      u'computer': [u'processor_name', u'processor_brand', u'operating_system', u'dedicated_graphic_memory_capacity', u'system_memory'],
    #      u'cpu': [u'processor_name', u'operating_system', u'graphics_memory', u'ram', u'memory_technology'],
    #      u'dslr_camera': [u'sensor_type', u'video_quality']}


class AttributesIndexer(object):
    def __init__(self, cms, df, attrs_for_cms):
        self.cms = cms
        self.df = df
        with open('comps/attribute_extraction/function_files/attributes_dict.json') as json_file:
            self.cms_attributes_definitions_dict = {k: v for k, v in json.load(json_file).iteritems() if k in attrs_for_cms}
        self.trim_df_with_needed_columns_from_db(self.df)
        self.attributes_columns = self.get_attributes_columns(self.df)
    
    def run(self):
        self.df['attrs_indexes_dict'] = self.df.apply(self.annotate_row_with_indexes_of_attr_values, axis=1)
        self.remove_rows_with_overlapping_attributes()
        fn = 'tmp/with_indexes_{ts}.csv'.format(ts=time_string())
        df_to_csv(self.df, fn)
        print fn
    
    def annotate_row_with_indexes_of_attr_values(self, row):
        d = dict()
        for attr_name in self.attributes_columns:
            doc = row.product_name
            doc = doc.lower()
            possible_values_of_attribute_in_json = list(itertools.chain.from_iterable([syns_joined_with_coma.split(',')
                                                                                       for syns_joined_with_coma in self.cms_attributes_definitions_dict.get(attr_name, [])]))
            if not possible_values_of_attribute_in_json:
                print 'warning:', attr_name, 'not in json'
                continue  # todo
            for possible_value in possible_values_of_attribute_in_json:
                possible_value = possible_value.lower().strip()
                if possible_value == '':
                    continue
                if possible_value in doc:
                    value_start_index_in_doc = doc.index(possible_value)
                    value_end_index_in_doc = value_start_index_in_doc + len(possible_value)
                    d[attr_name] = value_start_index_in_doc, value_end_index_in_doc
        return d
    
    def trim_df_with_needed_columns_from_db(self, df):
        if 'description' not in df.columns:
            self.df = DataFetchFuncs.annotate_with_product_data(df, ['description'])
        self.df = self.df.rename(columns={col: col.replace(' ', '_') for col in self.df.columns})
    
    @staticmethod
    def get_attributes_columns(df):
        return [col for col in df.columns if col not in ['product_id', 'product_name', 'description', 'product_type', ]]
    
    def remove_rows_with_overlapping_attributes(self):
        self.df['overlaps'] = self.df['attrs_indexes_dict'].apply(dict_of_tuples_with_overlapping_indexes)
        self.df=self.df[self.df['overlaps'].astype(str)=='[]']


def trim(x):
    return x.replace(' ', '_').strip().lower()


class SpacyClassifier(object):
    def __init__(self, training_df, cms, output_dir):
        self.cms = cms
        self.output_dir = output_dir
        self.training_df = training_df
        self.full_formatted_data = self.prepare_spacy_input()
        text, labels = zip(*self.full_formatted_data)
        self.text_train, self.text_test, \
        self.labels_train, self.labels_test = train_test_split(text, labels, test_size=0.2, random_state=42)
    
    def prepare_spacy_input(self):
        # specific to spacy input, for each tested model need to adjust format accordingly.
        import ast
        self.training_df = self.training_df[self.training_df['attrs_indexes_dict'] != u'{}']
        input_data = []
        for _, row in self.training_df.iterrows():
            raw_dict = ast.literal_eval(row['attrs_indexes_dict'])
            ent_data = [(raw_dict[key] + (key,)) for key in raw_dict.keys()]
            input_data.append((row['product_name'], {"entities": ent_data}))
        return input_data
    
    def train_spacy(self):
        # Train model
        nlp = English()
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        for _, annotations in self.full_formatted_data:
            for ent in annotations['entities']:
                ner.add_label(ent[2])
        optimizer = nlp.begin_training()
        for itn in range(10):
            print "Starting iteration # " + str(itn)
            train_set = zip(self.text_train, self.labels_train)
            random.shuffle(train_set)
            losses = {}
            batches = minibatch(train_set, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.2, sgd=optimizer, losses=losses)
            print "Losses: " + str(losses)
        # Save model
        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.meta['name'] = "spaCy ner model - cms '%s'" % self.cms
            nlp.to_disk(output_dir)
            print 'Model saved to ', output_dir
    
    def evaluate_model(self):
        scorer = Scorer()
        nlp = spacy.load(self.output_dir)
        test_set = zip(self.text_test, self.labels_train)
        for text, annot in test_set:
            doc_gold_text = nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot['entities'])
            pred_value = nlp(text)
            scorer.score(pred_value, gold)
        return scorer.scores
    
    def evaluate_example(self, test_text):
        nlp = spacy.load(self.output_dir)
        doc = nlp(test_text)
        print "Entities detected in '%s':" % test_text
        for ent in doc.ents:
            print ent.label_, ent.text


def get_product_tokens_and_tags(product_doc, att_dict):
    tokens, tags = [], []
    for word in product_doc.strip().split():
        tokens.append(word)
        tag = [att if att_dict[att][0] == product_doc.index(word) else 'O' for att in att_dict.keys()][0]
        tags.append(tag)
    return tokens, tags


class BiLSTMClassifier(object):
    def __init__(self, df):
        self.df = df
        self.tokens, self.tags = self.prepare_input()

    def prepare_input(self):
        import ast
        all_tokens = []
        all_tags = []
        self.df = self.df[self.df['attrs_indexes_dict'] != u'{}']
        for _, row in self.df.iterrows():
            product_doc = row['product_name']
            att_dict = ast.literal_eval(row['attrs_indexes_dict'])
            tokens, tags = get_product_tokens_and_tags(product_doc, att_dict)
            all_tokens.append(tokens)
            all_tags.append(tags)
        return all_tokens, all_tags

    # todo: can't import seaborn/matplotlib, apt-get install python-tk did not work; need to solve this for
    #  visualisations
    def sentence_length_distribution_analysis(self):
        lengths = list(map(lambda x: len(x), self.df['product_name']))
        ax = sns.distplot(lengths)
        ax.set(xlabel="Number of tokens in a sentences", ylabel="% of sentences")
        print "Median: ", np.median(lengths)
        print "Average: ", round(np.mean(lengths), 2)

    def summary(self):
        flat_list = list(itertools.chain.from_iterable(self.tags))
        cnt = Counter()
        for word in flat_list:
            cnt[word] += 1
        count_dict = dict(cnt)
        count_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        print("Number of unique items: ", len(count_items))
        print("Average count: ", round(len(self.tags) / len(count_items)), "\n")
        total_items = len(self.tags)
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
        print 'test'

    def tag_presence_percent(self):
        pass

def main():
    
    fn_indexed = 'tmp/with_indexes_200719-182932.csv'
    df_indexed = pd.read_csv(fn_indexed, encoding='utf-8')
    b = SpacyClassifier(df_indexed, 'computer', '../tmp')
    b.train_spacy()
    b.evaluate_model()
    print 'tse'

import zipp
def main_create_inedexed():
    fn = '/tmp/computer_1594228867.csv'
    df = pd.read_csv(fn, encoding='utf-8')
    cms = 'computer'
    cms_to_relevant_attribute_names = map_cms_to_relevant_attribute_names()
    attrs_for_cms = cms_to_relevant_attribute_names[cms]
    a = AttributesIndexer(cms, df, attrs_for_cms)
    a.run()
