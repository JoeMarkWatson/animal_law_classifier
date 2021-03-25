import pandas as pd
import numpy as np
import tensorflow_hub as hub
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
import re
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold  # more applicable than Kfold as unbalanced classes: http://ethen8181.github.io/machine-learning/model_selection/model_selection.html#K-Fold-Cross-Validation
from keras import Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from keras.constraints import maxnorm
from joblib import dump, load

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

def remove_punct(text):
    """remove some common punctuation, including full stops (which means that this col cannot be used for embedding"""
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_©~'''  # added © after investigation
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")
    return text

regex = re.compile(r'[\n\r\t\xa0\x0c]')
def remove_first_n_words(text):
    """remove first n words, as they frequently contain html code and judges names"""
    text = re.sub("(\w+)", "", text, 200)  # selecting 200
    text = regex.sub("", text)
    return text

english_words = set(nltk.corpus.words.words())  # and remove jury from here
jury_words = {'jury', 'juries', 'jurying', 'juried', 'juror'}
def retain_english(text):
    """drop any non-English words"""
    eng_text = []
    text_words = nltk.word_tokenize(text)
    for word in text_words:
        if word in english_words:  # https://stackoverflow.com/questions/41290028/removing-non-english-words-from-text-using-python
            if word not in jury_words:
                eng_text.append(word)
                eng_text.append(" ")
    return "".join(eng_text)

dd['jtfc'] = dd['judgment_text'].map(str) \
                            .map(remove_urls) \
                            .map(remove_html_tags) \
                            .map(lambda x: x.lower()) \
                            .map(lambda x: x.strip()) \
                            .map(lambda x: re.sub(r'\d+', '', x)) \
                            .map(remove_punct) \
                            .map(remove_first_n_words) \
                            .map(retain_english)  # takes approx 7 mins run time

##DATA SAVE POINT
#dd.to_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text.csv', index=False)  # old stemmed text
#dd.to_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text_23Mar.csv', index=False)
##DATA LOAD POINT
#dd = pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text_23Mar.csv')

# add jthc to your X_train selection
dd.columns = ['Link', 'case_name', 'year', 'classification_narrow', 'word_count_pre_stem', 'judgment_text', 'jtfc']  # to
# rename 'link'
X_train = pd.merge(X_train, dd, how="inner")
X_test = pd.merge(X_test, dd, how="inner")


# # # USE embeddings model prep

# load USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
embed = hub.load(module_url)

all_mean_embs = pd.DataFrame()  # takes approx 2.5 hrs
for jt in dd['judgment_text']:
    m = 0
    n = 0
    wt_list = []
    wt_list.append(sent_tokenize(jt))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()
    print("Original judgment sentence count is " + str(len(wt_list_vals)))
    while m < 199:
        m += len(wt_list_vals[n].split())
        n += 1
    wt_list_vals = [wt for wt in wt_list_vals[n:] if len(wt) < 1000]
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
###all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')  # written 08.01.2021
#all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/USE_embeddings24Mar.csv')  # written 23.03.2021
# DATA LOADING POINT BELOW
###all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')
#all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/USE_embeddings24Mar.csv')
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

all_mean_embs = pd.DataFrame()  # takes approx 8 hrs
for jt in dd['judgment_text']:
    m = 0
    n = 0
    wt_list = []
    wt_list.append(sent_tokenize(jt))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()
    print("Original judgment sentence count is " + str(len(wt_list_vals)))
    while m < 199:
        m += len(wt_list_vals[n].split())
        n += 1
    wt_list_vals = [wt for wt in wt_list_vals[n:] if len(wt) < 1000]
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
#all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/baseBERT_embeddings22Mar.csv')
# DATA LOADING POINT BELOW
##all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/baseBERT_embeddings1Mar.csv')
#all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/baseBERT_embeddings22Mar.csv')
#all_mean_embs = all_mean_embs.iloc[:, 1:]

sBERT_embs = pd.concat([all_mean_embs, df['Link'].reset_index(drop=True)], axis=1)  # adding link onto end of sBERT_embs
X_train_sBERT_embs = pd.merge(X_train['Link'], sBERT_embs, how='inner')
X_test_sBERT_embs = pd.merge(X_test['Link'], sBERT_embs, how='inner')


# # # model 1: tuned linearSVC with tuned tfidfVectorizer for jtfc

# Showing that stemming can be tuned through the tokenizer option: https://gist.github.com/deargle/b57738c8ce2b4ed6ca90f86d5422431f
# Showing that it is OK to lemma before n-gram selection (and perhaps preferable to allow this as an option, or even
# just carry it out before tuning): https://stackoverflow.com/questions/47219389/compute-word-n-grams-on-original-text-or-after-lemma-stemming-process
# When lemmatizing, you need to show context (and pos='v' is likely fine for this): https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

def lemmatizer(text):
    words = [word for word in nltk.word_tokenize(text) if len(word) > 1]  # if len(word) > 1 to retain words 2+ characters long
    lemmas = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in words]
    return lemmas

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='word')),
    ('clf', LinearSVC())
])  # https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8 for rel deep explanation of SVMs

parameters = {

    'tfidf__tokenizer': (lemmatizer, None),
    # 'tfidf__stop_words': (None, 'english'),  # used by Medvedeva, but you are lemmatizing so this does not apply
    # cleanly and you do similar with max_df (which Medvedeva does not use)
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': (0.6, 0.7, 0.8),  # ignore ngrams that occur as more than .X of corpus
    'tfidf__min_df': (1, 2, 3),  # ignore ngrams featuring in less than 1, 2, 3 documents
    # 'tfidf__use_idf': (False, True),  # not tempted by this, but used by Medvedeva
    # 'tfidf__binary': (False, True),  # not tempted by this, but used by Medvedeva
    # 'tfidf__norm': (None, 'l1', 'l2'),  # not tempted by this, but used by Medvedeva
    'tfidf__max_features': (1000, 2000),  # Medvedeva permitted uncapped but this likely helps here given limited data

    'clf__C': (0.1, 1, 5)

}

cv = RandomizedSearchCV(pipeline, param_distributions=parameters, cv=StratifiedKFold(5), n_iter=300, n_jobs=-1, verbose=1, random_state=1)
t0 = time()
cv.fit(X_train['jtfc'], y_train)
print("tfidf model tuned in %0.2f mins" % ((time() - t0)/60))  # to show total time taken for training
# note that run time increases approx 1 min for each additional n_iter
print("Best accuracy: {}\nBest combination: {}".format(cv.best_score_, cv.best_params_))  # Medvedeva art: "For most
# articles unigrams achieved the highest results"
with open("/Users/joewatson/Desktop/animal_law_classifier/tfidf_score_params.txt", "w") as text_file:
    text_file.write("Best accuracy: {}\nBest combination: {}".format(cv.best_score_, cv.best_params_))

final_model = cv.best_estimator_

# # save model
#loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf.hdf5"  # added 'new_' to ensure already-run models are kept
#dump(cv.best_estimator_, loc_string)
# # load model
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf.hdf5')

y_pred = final_model.predict(X_test['jtfc'])  # note you cannot predict proba for LinearSVC unless further
# work: https://tapanpatro.medium.com/linearsvc-doesnt-have-predict-proba-ed8f48f47c55
print("Accuracy: {}".format(final_model.score(X_test['jtfc'], y_test)))
print(classification_report(y_test, y_pred))

# Below draws from: https://towardsdatascience.com/how-to-get-feature-importances-from-any-sklearn-pipeline-167a19f1214
# And the following source could also be checked:
# https://towardsdatascience.com/extracting-plotting-feature-names-importance-from-scikit-learn-pipelines-eb5bfa6a31f4
feature_names = final_model.named_steps['tfidf'].get_feature_names()
coefs = final_model.named_steps["clf"].coef_.flatten()
zipped = zip(feature_names, coefs)
df = pd.DataFrame(zipped, columns=["feature", "value"])
df["abs_value"] = df["value"].apply(lambda x: abs(x))
df = df.sort_values("value", ascending=False)  # note the presence of animal in the list, which is there as 'animal'
# removed from some training set judgments as it was just in the first 200 words of the judgment in each case

# plot features that
import matplotlib.pyplot as plt
df_upper = df.head(10)
df_upper = df_upper.sort_values("value", ascending=True)
plt.bar(x=df_upper['feature'],
        height=df_upper['abs_value'],
        color='steelblue')
plt.xticks(rotation=90)
plt.ylabel("absolute coefficient value")
plt.title("Features most predictive of animal protection law")
plt.show()

df_lower = df.tail(10)
df_lower = df_lower.sort_values("abs_value", ascending=False)
plt.bar(x=df_lower['feature'],
        height=df_lower['abs_value'],
        color='indianred')
plt.xticks(rotation=90)
plt.ylabel("absolute coefficient value")
plt.title("Features most predictive of not animal protection law")
plt.show()


# # # models 2 and 3: tuned linearSVC for USE and tuned linearSVC for sBERT

data_dict = {
                'use_sets': [X_train_USE_embs, X_test_USE_embs],
                'sbert_sets': [X_train_sBERT_embs, X_test_sBERT_embs]
}  # creating a data_dict using the pre-created (or, pre-loaded) embeddings

svm = LinearSVC(max_iter=10000)  # increasing max_iter (suggested by convergence fail error message when running sBERT,
# and one of the multiple poss options given here: https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
params = {'C': (0.1, 1, 5)}  # gamma not applicable to LinearSVC
grid_search = GridSearchCV(svm, params, cv=StratifiedKFold(5), n_jobs=-1, verbose=1)
for d_d in data_dict:
    grid_result = grid_search.fit(data_dict[d_d][0].iloc[:, 1:], y_train)
    print("Tuned model parameters: {}".format(grid_result.best_params_))
    print("Average tuned model cv score: {}".format(grid_result.best_score_))
    # implement on test set (and cannot predict proba for LinearSVC unless further work: https://tapanpatro.medium.com/linearsvc-doesnt-have-predict-proba-ed8f48f47c55)
    y_pred = (grid_result.predict(data_dict[d_d][1].iloc[:, 1:]) > 0.5).astype("int32")
    print("Accuracy: {}".format(grid_result.score(data_dict[d_d][1].iloc[:, 1:], y_test)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
    # print(f1_score(y_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/33326810/scikit-weighted-f1-score-calculation-and-usage
    print(f1_score(y_pred, y_test, average="macro"))
    print(accuracy_score(y_pred, y_test))


# # # models 4 and 5: tuned keras for USE and tuned keras for sBERT

# if playing with tfidf vector (tuned for sklearn linearSVC), then run the following:
#vectorizer = final_model.named_steps['tfidf']
#X_train_tfidf = vectorizer.transform(X_train['jtfc']).toarray()  # https://stackoverflow.com/questions/62871108/error-with-tfidfvectorizer-but-ok-with-countvectorizer
#X_train_tfidf = pd.concat([X_train['Link'].reset_index(drop=True), pd.DataFrame(X_train_tfidf)], axis=1)
#X_test_tfidf = vectorizer.transform(X_test['jtfc']).toarray()
#X_test_tfidf = pd.concat([X_test['Link'].reset_index(drop=True), pd.DataFrame(X_test_tfidf)], axis=1)

# define create_model
def create_model(learning_rate, activation, dense_nparams, dropout):  # leaving no. of layers and layer order constant
    opt = Adam(lr=learning_rate)  # create an Adam optimizer with the given learning rate
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(max_f,)))  # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model.add(Dense(dense_nparams, activation=activation, kernel_constraint=maxnorm(dropout*15)))  # create input layer
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))  # create output layer
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model with optimizer, loss, and metrics
    return model

# create a test-set and train-set dict
data_dict = {
    'USE_embs': [X_train_USE_embs, X_test_USE_embs],
    'sBERT_embs': [X_train_sBERT_embs, X_test_sBERT_embs] # ,
    #'tfidf': [X_train_tfidf, X_test_tfidf],  # tfidf vectors not used here, but note https://stackoverflow.com/questions/62871108/error-with-tfidfvectorizer-but-ok-with-countvectorizer
}

#d_d = [X_train_tfidf, X_test_tfidf]  # temporarily here to enable quick re-running of specific embs
for d_d in data_dict:

    max_f = data_dict[d_d][0].shape[1]-1
    #max_f = d_d[0].shape[1] - 1

    param_grid = {
        'epochs': [20, 50, 100, 200],
        'dense_nparams': [max_f / 16, max_f / 8, max_f / 4, max_f / 2],
        #'batch_size': [1, 5, 10],
        'learning_rate': [0.1, 0.01, 0.001],
        'activation': ['relu'],
        'dropout': [0.2, 0]  # DROPOUT REFS https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    }  # define param_grid after max_f so it uses the current max_f value

    model = KerasClassifier(build_fn=create_model)

    np.random.seed(1)  # attempt to max reproducibility...
    seed(1)  # from numpy again I think so poss duplicating
    tensorflow.random.set_seed(1)  # from tf
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, cv=StratifiedKFold(5), n_jobs=-1,
                                       n_iter=100, random_state=1)  # ... but cannot be fully reproducible as not single threaded: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development  https://datascience.stackexchange.com/questions/37413/why-running-the-same-code-on-the-same-data-gives-a-different-result-every-time
                                        # increased n_iter from 10 to 100 for final modelling
    # ran_result = random_search.fit(d_d[0].iloc[:, 1:], y_train, verbose=0)  # takes 5-10 minutes
    ran_result = random_search.fit(data_dict[d_d][0].iloc[:, 1:], y_train, verbose=0)  # takes 5-10 minutes
    print("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "/Users/joewatson/Desktop/animal_law_classifier/" + d_d + "_score_params.txt"
    with open(loc_string, "w") as text_file:
        text_file.write("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_" + d_d + ".hdf5"  # added 'new_' to ensure already-run models are kept
    #loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_temp.hdf5"
    ran_result.best_estimator_.model.save(loc_string)

    best_model = tensorflow.keras.models.load_model(loc_string)  # https://www.tensorflow.org/api_docs/python/tf/
    # keras/models/load_model  # compile=True appears not to be required

    y_pred_proba = best_model.predict(data_dict[d_d][1].iloc[:, 1:])  # in case prediction info is required
    #y_pred_proba = best_model.predict(d_d[1].iloc[:, 1:])  # in case prediction info is required
    y_pred = (best_model.predict(data_dict[d_d][1].iloc[:, 1:]) > 0.5).astype("int32")
    #y_pred = (best_model.predict(d_d[1].iloc[:, 1:]) > 0.5).astype("int32")

    print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
    print(classification_report(y_pred, y_test))
    # print(f1_score(y_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/33326810/scikit-weighted-f1-score-calculation-and-usage
    print(f1_score(y_pred, y_test, average="macro"))
    print(accuracy_score(y_pred, y_test))


# # #

# show base model performance (i.e., everything as class 1)
bool_pred = (y_test + 1) / (y_test + 1)
print(confusion_matrix(bool_pred, y_test, labels=[1, 0]))
print(classification_report(bool_pred, y_test))
# print(f1_score(bool_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/37358496/is-f1-micro-the-same-as-accuracy
print(f1_score(bool_pred, y_test, average="macro"))
print(accuracy_score(bool_pred, y_test))


# # # permutation tests - to be added

# # # links to already-run models

# loc_string = '/Users/joewatson/Desktop/LawTech/ran_search_strat_model_tfidf.hdf5'
# loc_string = '/Users/joewatson/Desktop/LawTech/ran_search_strat_model_sBERT_embs.hdf5'
# loc_string = '/Users/joewatson/Desktop/LawTech/ran_search_strat_model_USE_embs.hdf5'
# best_model = tensorflow.keras.models.load_model(loc_string)  # to load the selected model

# weights = best_model.layers[0].get_weights()[0]
# biases = best_model.layers[0].get_weights()[1]
