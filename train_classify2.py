import pandas as pd
import numpy as np
import tensorflow_hub as hub
import re
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from urllib.request import Request
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)  # https://stackoverflow.com/questions/58638701/importerror-cannot-import-name-set-random-seed-from-tensorflow-c-users-po
import random
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
porter = PorterStemmer()

# # # always run the below, even if using a later data loading point

# import labelled spreadsheet
df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled_h.csv")  # import same csv with Heathcote error fixed
df = df[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
df = df[df['Classification'] >= 0]  # retain labelled judgments only
df['classification_narrow'] = np.where(df['Classification'] == 1, 1, 0)  # make a narrow class column, covering animal protection law only

# train test split
X, y = df[['Link']], df['classification_narrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

#_______________________________________________________________________________________________________________________


# # # judgment gathering # # #

# make a df with Link as key
link_dict = df.set_index('Link').T.to_dict('list')
# demo how the list works
print(link_dict['https://www.bailii.org/uk/cases/UKPC/2000/47.html'][0])  # case_name
print(link_dict['https://www.bailii.org/uk/cases/UKPC/2000/47.html'][1])  # year
print(link_dict['https://www.bailii.org/uk/cases/UKPC/2000/47.html'][4])  # classification_narrow

# extract case text from list of Links, into a df comprised of link, case, year, classification, case_text
d = []
for l in link_dict.keys():
    req = Request(l, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html5lib")
    case_text = page_soup.get_text()  # scrape no more specif due to varying page layouts
    case_text = re.sub('\n', ' ', case_text)  # replace '\n' with ' '
    case_text = re.sub('@media screen|@media print|#screenonly|BAILII|Multidatabase Search|World Law', '', case_text)  # remove some patterns
    d.append(
        {
            'link': l,
            'case_name': link_dict[l][0],
            'year': link_dict[l][1],
            'classification_narrow': link_dict[l][4],
            'word_count_pre_stem': len(word_tokenize(case_text)),  # included here to help catch scrape error
            'judgment_text': case_text
        }
    )
print("done")

dd = pd.DataFrame(d)

##DATA SAVE POINT
#dd.to_csv('/Users/joewatson/Desktop/LawTech/scraped_500_text.csv', index=False)
#dd.equals(pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_text.csv'))  # shows save kept all info
##DATA LOAD POINT
#dd = pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_text.csv')


# # # tfidf model prep

# create a jtfc (judgment text further cleaning) column, following cleaning advice on:
# https://medium.com/@am.benatmane/keras-hyperparameter-tuning-using-sklearn-pipelines-grid-search-with-cross-
# validation-ccfc74b0ce9f
def remove_punct(text):
    """remove some common punctuation, including full stops (which means that this col cannot be used for embedding"""
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")
    return text

def remove_urls(text):
    """remove hypertext links"""
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftp?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return text

def remove_html_tags(text):
    """remove html tags"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

regex = re.compile(r'[\n\r\t\xa0\x0c]')
def remove_first_n_words(text):
    """remove first n words, as they frequently contain html code and judges names"""
    text = re.sub("(\w+)", "", text, 200)  # selecting 200
    text = regex.sub("", text)
    return text

punctuations="?:!.,;"
english_words = set(nltk.corpus.words.words())
def stem_words(text):
    """stem each judgment, word-by-word, and drop any non-English words and remaining punctuation"""
    stem_text = []
    text_words = nltk.word_tokenize(text)
    for word in text_words:
        if word in punctuations:
            text_words.remove(word)
        if word in english_words:  # https://stackoverflow.com/questions/41290028/removing-non-english-words-from-text-using-python
            stem_text.append(porter.stem(word))
            stem_text.append(" ")
    return "".join(stem_text)  # https://www.datacamp.com/community/tutorials/stemming-lemmatization-python


dd['jtfc'] = dd['judgment_text'].map(str) \
                            .map(remove_urls) \
                            .map(remove_html_tags) \
                            .map(lambda x: x.lower()) \
                            .map(lambda x: x.strip()) \
                            .map(lambda x: re.sub(r'\d+', '', x)) \
                            #.map(remove_html_tags) \  # DUPLICATE WAS THERE IN RUN MODEL
                            .map(remove_first_n_words) \
                            .map(remove_punct) \
                            .map(stem_words)  # takes approx 8 mins run time

##DATA SAVE POINT
#dd.to_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text.csv', index=False)
#dd.equals(pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text.csv'))  # shows save kept all info
##DATA LOAD POINT
#dd = pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text.csv')

# add jthc to your X_train selection
dd.columns = ['Link', 'case_name', 'year', 'classification_narrow', 'word_count_pre_stem', 'judgment_text', 'jtfc']
X_train = pd.merge(X_train, dd, how="inner")
X_test = pd.merge(X_test, dd, how="inner")

from sklearn.model_selection import RandomizedSearchCV
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold  # more applicable than Kfold as unbalanced classes: http://ethen8181.github.io/machine-learning/model_selection/model_selection.html#K-Fold-Cross-Validation
from keras.layers import Dense

max_f = 768
vectorizer = TfidfVectorizer(max_df=0.8, max_features=max_f, ngram_range=(1, 1))  # Medvedeva art: "For most articles
# unigrams achieved the highest results"
vectorizer.fit(X_train['jtfc'])  # quick to run, so use load point from above
X_train_tfidf = vectorizer.transform(X_train['jtfc']).toarray()  # https://stackoverflow.com/questions/62871108/error-with-tfidfvectorizer-but-ok-with-countvectorizer
X_train_tfidf = pd.concat([X_train['Link'].reset_index(drop=True), pd.DataFrame(X_train_tfidf)], axis=1)

X_test_tfidf = vectorizer.transform(X_test['jtfc']).toarray()
X_test_tfidf = pd.concat([X_test['Link'].reset_index(drop=True), pd.DataFrame(X_test_tfidf)], axis=1)


# # # USE embeddings model prep

# load USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
embed = hub.load(module_url)

all_mean_embs = pd.DataFrame()
for jt in dd['judgment_text']:
    wt_list = []
    wt_list.append(sent_tokenize(jt))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()
    print("Original judgment sentence count is " + str(len(wt_list_vals)))
    wt_list_vals = [wt for wt in wt_list_vals if len(wt) < 1000]  # changed to 1000 from 2000
    if len(wt_list_vals) > 5000:
        random.seed(1)
        wt_list_vals = random.sample(wt_list_vals, 5000)
    X_embed = embed(wt_list_vals)
    my_array = [np.array(emb) for emb in X_embed]
    my_df = pd.DataFrame(my_array)
    means = []
    for c in my_df.columns:
        means.append(my_df[c].mean())
    means_df = pd.DataFrame(means).T
    all_mean_embs = pd.concat([all_mean_embs, means_df])
    print("Done " + str(len(all_mean_embs)) + " embedding averages")

# DATA SAVING POINT BELOW
#all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')  # written 08.01.2021
# DATA LOADING POINT BELOW
#all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')
#all_mean_embs = all_mean_embs.iloc[:, 1:]

USE_embs = pd.concat([all_mean_embs, df['Link'].reset_index(drop=True)], axis=1)  # adding link onto end of USE_embs
X_train_USE_embs = pd.merge(X_train['Link'], USE_embs, how='inner')
X_test_USE_embs = pd.merge(X_test['Link'], USE_embs, how='inner')


# # # BERT embeddings model prep

from sentence_transformers import SentenceTransformer
# load sBERT model
#model = SentenceTransformer('bert-large-nli-mean-tokens')  # outputs a 1024-dimensional vector
# large bert too slow, base bert being used to increase speed. (Appears approx 5x the speed at about 80 judgments/hour)
model = SentenceTransformer('bert-base-nli-mean-tokens')  # outputs a 768-dimensional vector

all_mean_embs = pd.DataFrame()
for jt in dd['judgment_text']:
    wt_list = []
    wt_list.append(sent_tokenize(jt))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()
    print("Original judgment sentence count is " + str(len(wt_list_vals)))
    wt_list_vals = [wt for wt in wt_list_vals if len(wt) < 1000]
    if len(wt_list_vals) > 5000:
        random.seed(1)
        wt_list_vals = random.sample(wt_list_vals, 5000)
    X_embed = model.encode(wt_list_vals)
    my_df = pd.DataFrame(X_embed)
    means = []
    for c in my_df.columns:
        means.append(my_df[c].mean())
    means_df = pd.DataFrame(means).T
    all_mean_embs = pd.concat([all_mean_embs, means_df])
    print("Done " + str(len(all_mean_embs)) + " embedding averages")

# DATA SAVING POINT BELOW
#all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/baseBERT_embeddings1Mar.csv')
# DATA LOADING POINT BELOW
#all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/baseBERT_embeddings1Mar.csv')
#all_mean_embs = all_mean_embs.iloc[:, 1:]

sBERT_embs = pd.concat([all_mean_embs, df['Link'].reset_index(drop=True)], axis=1)  # adding link onto end of sBERT_embs
X_train_sBERT_embs = pd.merge(X_train['Link'], sBERT_embs, how='inner')
X_test_sBERT_embs = pd.merge(X_test['Link'], sBERT_embs, how='inner')


# # # model specification and tuning

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold  # more applicable than Kfold as unbalanced classes: http://ethen8181.github.io/machine-learning/model_selection/model_selection.html#K-Fold-Cross-Validation
from keras import Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# define create_model
def create_model(learning_rate, activation, dense_nparams):  # then add batch size later, leaving no. of layers and layer order constant
    opt = Adam(lr=learning_rate)  # create an Adam optimizer with the given learning rate
    model = Sequential()
    model.add(Dense(dense_nparams, input_shape=(max_f,), activation=activation))  # create input layer
    #model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))  # create output layer
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model with optimizer, loss, and metrics
    return model


# create a test-set and train-set dict
data_dict = {
    'tfidf': [X_train_tfidf, X_test_tfidf],
    'USE_embs': [X_train_USE_embs, X_test_USE_embs],
    'sBERT_embs': [X_train_sBERT_embs, X_test_sBERT_embs],
}


for d_d in data_dict:

    max_f = data_dict[d_d][0].shape[1]-1

    param_grid = {
        'epochs': [20, 50, 100, 200],
        'dense_nparams': [max_f / 16, max_f / 8, max_f / 4, max_f / 2],
        'batch_size': [1, 5, 10],
        'learning_rate': [0.1, 0.01, 0.001],
        'activation': ['relu']
        # 'dropout': [0.1, 0]
    }  # define param_grid after max_f so it uses the current max_f value

    model = KerasClassifier(build_fn=create_model)

    np.random.seed(1)  # attempt to max reproducibility...
    seed(1)  # from numpy again I think so poss duplicating
    tensorflow.random.set_seed(1)  # from tf
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, cv=StratifiedKFold(5), n_jobs=-1,
                                       random_state=1)  # ... but cannot be fully reproducible as not single threaded: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development  https://datascience.stackexchange.com/questions/37413/why-running-the-same-code-on-the-same-data-gives-a-different-result-every-time

    ran_result = random_search.fit(data_dict[d_d][0].iloc[:, 1:], y_train, verbose=0)  # takes 5-10 minutes
    print("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_" + d_d + ".hdf5"  # added 'new_' to ensure already-run models are kept
    ran_result.best_estimator_.model.save(loc_string)

    best_model = tensorflow.keras.models.load_model(loc_string)  # https://www.tensorflow.org/api_docs/python/tf/
    # keras/models/load_model  # compile=True appears not to be required

    y_pred_proba = best_model.predict(data_dict[d_d][1].iloc[:, 1:])  # in case prediction info is required
    y_pred = (best_model.predict(data_dict[d_d][1].iloc[:, 1:]) > 0.5).astype("int32")

    print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
    print(classification_report(y_pred, y_test))
    # print(f1_score(y_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/33326810/scikit-weighted-f1-score-calculation-and-usage
    print(f1_score(y_pred, y_test, average="macro"))
    print(accuracy_score(y_pred, y_test))


# show base model performance (i.e., everything as class 1)
bool_pred = (y_test + 1) / (y_test + 1)
print(confusion_matrix(bool_pred, y_test, labels=[1, 0]))
print(classification_report(bool_pred, y_test))
# print(f1_score(bool_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/37358496/is-f1-micro-the-same-as-accuracy
print(f1_score(bool_pred, y_test, average="macro"))
print(accuracy_score(bool_pred, y_test))

# # # links to already-run models

# loc_string = '/Users/joewatson/Desktop/LawTech/ran_search_strat_model_tfidf.hdf5'
# loc_string = '/Users/joewatson/Desktop/LawTech/ran_search_strat_model_sBERT_embs.hdf5'
# loc_string = '/Users/joewatson/Desktop/LawTech/ran_search_strat_model_USE_embs.hdf5'
# best_model = tensorflow.keras.models.load_model(loc_string)  # to load the selected model

# weights = best_model.layers[0].get_weights()[0]
# biases = best_model.layers[0].get_weights()[1]
