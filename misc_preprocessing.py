import nltk
import pandas as pd
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
def pos_tag(fn_of_data_with_indexes):
    df=pd.read_csv(fn_of_data_with_indexes,usecols=['product_name',	'attrs_indexes_dict'])
    text = nltk.word_tokenize("And now for something completely different")
    nltk.pos_tag(text)
if __name__ == '__main__':
    pos_tag('computer_with_indexes_200720-183555.csv')
