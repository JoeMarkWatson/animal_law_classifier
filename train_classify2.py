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

# fix error in test set
index_no = X_test.loc[lambda X_test: X_test['Link'] == 'https://www.bailii.org/ew/cases/EWHC/Admin/2002/908.html', :].index[0]
y_test[index_no] = 1

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
jury_words = {}
jury_words = {'jury', 'juries', 'jurying', 'juried', 'juror'}
#jury_words = {'jury', 'juries', 'jurying', 'juried', 'juror',
#              'schedule', 'scheduls', 'scheduling', 'scheduled', 'scheduler',
#              'investigation', 'investigations', 'investigating', 'investigated', 'investigator'}
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
#dd.to_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text_Apr_JuryIn.csv', index=False)
##DATA LOAD POINT
#dd = pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text_23Mar.csv')
#dd = pd.read_csv('/Users/joewatson/Desktop/LawTech/scraped_500_cleaned_text_Apr_JuryIn.csv')

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

    'tfidf__tokenizer': (lemmatizer, None),  # [(lemmatizer)]
    # 'tfidf__stop_words': (None, 'english'),  # used by Medvedeva, but you are lemmatizing so this does not apply
    # cleanly and you do similar with max_df (which Medvedeva does not use)
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # [(1, 1)]
    'tfidf__max_df': (0.6, 0.7, 0.8),  # [(0.6)] # ignore ngrams that occur as more than .X of corpus  # (0.6, 0.7, 0.8)
    'tfidf__min_df': (1, 2, 3),  # [(5)] # ignore ngrams featuring in less than 1, 2, 3 documents  # ([1])
    # 'tfidf__use_idf': (False, True),  # not tempted by this, but used by Medvedeva
    # 'tfidf__binary': (False, True),  # not tempted by this, but used by Medvedeva
    # 'tfidf__norm': (None, 'l1', 'l2'),  # not tempted by this, but used by Medvedeva
    'tfidf__max_features': (500, 1000),  # [(1000)] # Medvedeva permitted uncapped but this likely helps here given limited data

    'clf__C': (0.1, 1, 2, 5),  # [(1)]
    'clf__loss': ('hinge', 'squared_hinge')  # [('squared_hinge')]

}

cv = RandomizedSearchCV(pipeline, param_distributions=parameters, cv=StratifiedKFold(5), n_iter=800, n_jobs=-1,
                        verbose=1, random_state=1, scoring='f1_macro', refit='f1_macro')  # to potentially report F1: https://scikit-learn.org/stable/modules/model_evaluation.html
# return_train_score=True gives by-split score results
t0 = time()
cv.fit(X_train['jtfc'], y_train)
print("tfidf model tuned in %0.2f mins" % ((time() - t0)/60))  # to show total time taken for training
# note that run time increases approx 1 min for each additional n_iter
print("Best f1_macro: {}\nBest combination: {}".format(cv.best_score_, cv.best_params_))  # Medvedeva art: "For most
# articles unigrams achieved the highest results"
#with open("/Users/joewatson/Desktop/animal_law_classifier/tfidf_score_params14Apr_5to1_jury_in.txt", "w") as text_file:
#    text_file.write("Best f1_macro: {}\nBest combination: {}".format(cv.best_score_, cv.best_params_))

final_model = cv.best_estimator_

# # save model
#loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf14Apr_5to1_jury_in.hdf5"  # added 'new_' to ensure already-run models are kept
#dump(cv.best_estimator_, loc_string)

# # load model
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf.hdf5')  # tuned on accuracy, jury was out during training
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf7Apr.hdf5')   # tuned on f1, jury was out during training
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf7Apr_5to1.hdf5')  # f1 and 5-1k, jury was out during training
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf7Apr_1to2_no_inv.hdf5')  # f1, longer 'jury words' list out during training
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf14Apr_5to1_jury_in_hinge.hdf5')  # f1 and hinge loss only allowed 19 Apr, jury in
#final_model = load('/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidf14Apr_5to1_jury_in.hdf5')  # f1, jury in  # the selected model

y_pred = final_model.predict(X_test['jtfc'])  # note you cannot predict proba for LinearSVC unless further
# work: https://tapanpatro.medium.com/linearsvc-doesnt-have-predict-proba-ed8f48f47c55
print("Accuracy: {}".format(final_model.score(X_test['jtfc'], y_test)))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
print(f1_score(y_pred, y_test, average="macro"))
print(accuracy_score(y_pred, y_test))

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

pos_neg_df = pd.concat([df_lower, df_upper]).reset_index(drop=True)  # pos and neg on one graphic
pos_neg_df['p_n_pred'] = np.where(pos_neg_df['value'] > 0.5, '#8bb5bb', '#b3a6d0')  # 'steelblue', 'indianred'
plt.bar(x=pos_neg_df['feature'],
        height=pos_neg_df['value'],
        color=pos_neg_df['p_n_pred'])
plt.xticks(rotation=90)
plt.ylabel("coefficient value")
plt.title("Most positive and negative lemma coefficient values")
plt.show()


# # # models 2 and 3: tuned linearSVC for USE and tuned linearSVC for sBERT

data_dict = {
                'use_sets': [X_train_USE_embs, X_test_USE_embs],
                'sbert_sets': [X_train_sBERT_embs, X_test_sBERT_embs]
}  # creating a data_dict using the pre-created (or, pre-loaded) embeddings

svm = LinearSVC(max_iter=10000)  # increasing max_iter (suggested by convergence fail error message when running sBERT,
# and one of the multiple poss options given here: https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
params = {'C': (0.1, 1, 2, 5, 10),
          'loss': ('hinge', 'squared_hinge')}  # gamma not applicable to LinearSVC

grid_search = GridSearchCV(svm, params, cv=StratifiedKFold(5), n_jobs=-1, verbose=1, scoring='f1_macro', refit='f1_macro')
for d_d in data_dict:
    grid_result = grid_search.fit(data_dict[d_d][0].iloc[:, 1:], y_train)
    print("Best f1_macro: {}\nBest combination: {}".format(grid_result.best_score_, grid_result.best_params_))
    # save model and best params
    loc_string = "/Users/joewatson/Desktop/animal_law_classifier/sk_" + d_d + "_score_params20Apr_hinge.txt"  # for saving hinge only
    with open(loc_string, "w") as text_file:
        text_file.write("Best macro_f1: {}\nBest combination: {}".format(grid_result.best_score_, grid_result.best_params_))
    loc_string = "/Users/joewatson/Desktop/LawTech/new_search_sk_" + d_d + "20Apr_hinge.hdf5"  # for saving hinge only
    dump(grid_search.best_estimator_, loc_string)
    # load model
    #grid_result = load('/Users/joewatson/Desktop/LawTech/new_search_sk_USE_sets7Apr.hdf5')
    ##grid_result = load('/Users/joewatson/Desktop/LawTech/new_search_sk_sBERT_sets7Apr.hdf5')  # squared hinge - outperformed by hinge
    #grid_result = load('/Users/joewatson/Desktop/LawTech/new_search_sk_sbert_sets20Apr_hinge.hdf5')  # hinge
    # implement on test set (and cannot predict proba for LinearSVC unless further work: https://tapanpatro.medium.com/linearsvc-doesnt-have-predict-proba-ed8f48f47c55)
    y_pred = (grid_result.predict(data_dict[d_d][1].iloc[:, 1:]) > 0.5).astype("int32")
    #print("macro_f1: {}".format(grid_result.score(data_dict[d_d][1].iloc[:, 1:], y_test)))
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
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model with optimizer, loss, and metrics  # to maybe get F1: https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
    return model

# create a test-set and train-set dict
data_dict = {
    'USE_embs': [X_train_USE_embs, X_test_USE_embs],
    'sBERT_embs': [X_train_sBERT_embs, X_test_sBERT_embs]  # ,
    #'tfidf': [X_train_tfidf, X_test_tfidf],  # tfidf vectors not used here, but note https://stackoverflow.com/questions/62871108/error-with-tfidfvectorizer-but-ok-with-countvectorizer
}

#d_d = [X_train_USE_embs, X_test_USE_embs]  # temporarily here to enable quick re-running of specific embs
#d_d = [X_train_sBERT_embs, X_test_sBERT_embs]  # temporarily here to enable quick re-running of specific embs
for d_d in data_dict:

    max_f = data_dict[d_d][0].shape[1]-1
    #max_f = d_d[0].shape[1] - 1

    param_grid = {
        'epochs': [10, 20, 50, 100],
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
                                       n_iter=400, random_state=1, scoring='f1_macro', refit='f1_macro')  # ... but cannot be fully reproducible as not single threaded: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development  https://datascience.stackexchange.com/questions/37413/why-running-the-same-code-on-the-same-data-gives-a-different-result-every-time
                                        # increased n_iter from 10 to 100 for final modelling
    # ran_result = random_search.fit(d_d[0].iloc[:, 1:], y_train, verbose=0)  # takes 5-10 minutes
    ran_result = random_search.fit(data_dict[d_d][0].iloc[:, 1:], y_train, verbose=0)  # takes 5-10 minutes
    print("Best f1_macro: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "/Users/joewatson/Desktop/animal_law_classifier/" + d_d + "_score_params7Apr.txt"
    with open(loc_string, "w") as text_file:
        text_file.write("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_" + d_d + "7Apr.hdf5"  # added 'new_' to ensure already-run models are kept
    # loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_USE_embs7Apr.hdf5"
    # loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_sBERT_embs7Apr.hdf5"
    #ran_result.best_estimator_.model.save(loc_string)  # include line to save trained model

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


# # # model 6: tuned keras for tuned TF-IDF

# same random seeds as models 4 and 5, but likely good to re-state
np.random.seed(1)  # attempt to max reproducibility...
seed(1)  # from numpy again I think so poss duplicating
tensorflow.random.set_seed(1)  # from tf

best_score_sf = 0  # best score so far
q = 1

for lmt in (lemmatizer, None):
    for ngrams in [(1, 1), (1, 2)]:
        for max_d in (0.6, 0.7, 0.8):
            for min_d in (1, 2, 3):
                for max_f in (500, 1000):
                    print("starting vectorising")
                    vectorizer = TfidfVectorizer(max_features=max_f)  # tokenizer=lmt, ngram_range=ngrams, max_df=max_d, min_df=min_d,
                    X_train_tfidf = vectorizer.fit_transform(X_train['jtfc']).toarray()
                    X_train_tfidf = pd.concat([X_train['Link'].reset_index(drop=True), pd.DataFrame(X_train_tfidf)], axis=1)
                    print("vectorising completed")

                    d_d = X_train_tfidf

                    max_f = d_d.shape[1] - 1

                    param_grid = {
                            'epochs': [10, 20, 50, 100],
                            'dense_nparams': [max_f / 20, max_f / 10, max_f / 4, max_f / 2],  # some diff divisors from embeddings
                            'learning_rate': [0.1, 0.01, 0.001],
                            'activation': ['relu'],
                            'dropout': [0.2, 0]
                    }

                    model = KerasClassifier(build_fn=create_model)  # same create_model from embeddings

                    random_search = RandomizedSearchCV(model, param_distributions=param_grid, cv=StratifiedKFold(5), n_jobs=-1,
                                                       n_iter=400, random_state=1, scoring='f1_macro', refit='f1_macro')

                    ran_result = random_search.fit(d_d.iloc[:, 1:], y_train, verbose=0)
                    print("trained mlp model " + str(q) + " of 72")
                    q += 1

                    if ran_result.best_score_ > best_score_sf:
                        print(best_score_sf)
                        best_score_sf = ran_result.best_score_
                        best_params_sf = ran_result.best_params_  # best params so far
                        best_vect = str(vectorizer)
                        loc_string = "/Users/joewatson/Desktop/LawTech/new_ran_search_strat_model_tfidfMLP_29May.hdf5"
                        ran_result.best_estimator_.model.save(loc_string)  # include line to save trained model
                        params_loc_string = "/Users/joewatson/Desktop/animal_law_classifier/tfidfMLP_score_params29May.txt"
                        with open(params_loc_string, "w") as text_file:
                            text_file.write("Best validation score: {}\nBest TFIDF spec: {}\nBest mlp params: {}".format(best_score_sf, best_vect, best_params_sf))

                        print("saving done, making X_test vectors")
                        X_test_tfidf = vectorizer.transform(X_test['jtfc']).toarray()  # use vectorizer fitted to best performing model
                        X_test_tfidf = pd.concat([X_test['Link'].reset_index(drop=True), pd.DataFrame(X_test_tfidf)], axis=1)
                        print("made X_test vectors")

# and after all trialling, save best param.s/tfidf spec.s and do predictions on test set vectors

best_model = tensorflow.keras.models.load_model(loc_string)
best_model.best_params
y_pred_proba = best_model.predict(X_test_tfidf.iloc[:, 1:])  # in case prediction info is required
#y_pred_proba = best_model.predict(d_d[1].iloc[:, 1:])  # in case prediction info is required
y_pred = (best_model.predict(X_test_tfidf.iloc[:, 1:]) > 0.5).astype("int32")
#y_pred = (best_model.predict(d_d[1].iloc[:, 1:]) > 0.5).astype("int32")

print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
print(classification_report(y_pred, y_test))
# print(f1_score(y_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/33326810/scikit-weighted-f1-score-calculation-and-usage
print(f1_score(y_pred, y_test, average="macro"))
print(accuracy_score(y_pred, y_test))


#_______________________________________________________________________________________________________________________


# # # show base model performance (i.e., everything as class 1)
bool_pred = (y_test + 1) / (y_test + 1)
print(confusion_matrix(bool_pred, y_test, labels=[1, 0]))
print(classification_report(bool_pred, y_test))
# print(f1_score(bool_pred, y_test, average="weighted"))  # https://stackoverflow.com/questions/37358496/is-f1-micro-the-same-as-accuracy
print(f1_score(bool_pred, y_test, average="macro"))
print(accuracy_score(bool_pred, y_test))


# # # permutation tests

# precision, or 'what proportion of predicted Positives is truly Positive?' https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
# precision = tp/(tp + fp)

# recall, or 'what proportion of actual Positives is correctly classified?' (ibid)
# recall = tp/(tp + fn)

# per class f1 score
# per_class_f1_score = 2 * (precision * recall)/(precision + recall)  # calc.d separately for class 1 and class 0
# macro f1 score, or the simple arithmetic mean of per-class F1 scores: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

def macro_f1(preds_class_1, preds_class_0):
    tp_1 = preds_class_1.sum()
    fp_1 = len(preds_class_0) - preds_class_0.sum()
    fn_1 = len(preds_class_1) - preds_class_1.sum()
    if tp_1 == 0:
        precision_1, recall_1, f1_class_1 = 1, 1, 1
    else:
        precision_1 = tp_1/(tp_1 + fp_1)
        recall_1 = tp_1/(tp_1 + fn_1)
        f1_class_1 = 2 * (precision_1*recall_1) / (precision_1+recall_1)

    tp_0 = sum(preds_class_0)
    fp_0 = len(preds_class_1) - preds_class_1.sum()
    fn_0 = len(preds_class_0) - preds_class_0.sum()
    if tp_0 == 0:
        precision_0, recall_0, f1_class_0 = 0, 0, 0
    else:
        precision_0 = tp_0 / (tp_0 + fp_0)
        recall_0 = tp_0 / (tp_0 + fn_0)
        f1_class_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

    mac_f1 = (f1_class_1 + f1_class_0)/2

    return mac_f1

#11 April - below is using outdated numbers, should be:
#500-1k model 1:
#[[11  4]
# [ 5 80]]
# F1 = 0.828211490742508, accuracy = 0.91
# Best f1_macro: 0.7801661389866814 on validation folds

#Model 2
#[[11  3]
# [ 5 81]]

#Model 3
#[[ 9  7]
# [ 7 77]]

#Model 4	Model 5
#[[13 11]	[[12 13]
# [ 3 73]]	[ 4 71]]

# so, below needs updating

# class_1
actual_1 = [1]*17
base_pred_1 = [1]*17
tfidf_pred_1 = [1]*12 + [0]*5  # confusion matrix for tfidf model [[12  2] over [ 5 81]]
USE_sk_pred_1 = [1]*11 + [0]*6  # [[11  3] over [ 6 80]]
sBERT_sk_pred_1 = [1]*12 + [0]*5  # [[ 12  8] over [ 5 75]]
USE_keras_pred_1 = [1]*14 + [0]*3  # [[14  10] over [ 3 73]]
sBERT_keras_pred_1 = [1]*13 + [0]*4  # [[ 13  12] over [ 4 71]]
class_1_df = pd.DataFrame({'actual_1': actual_1, 'base_pred_1': base_pred_1,
                           'tfidf_pred_1': tfidf_pred_1, 'USE_sk_pred_1': USE_sk_pred_1,
                           'sBERT_sk_pred_1': sBERT_sk_pred_1, 'USE_keras_pred_1': USE_keras_pred_1,
                           'sBERT_keras_pred_1': sBERT_keras_pred_1})

# class_0 (not animal law)
actual_0 = [1]*83
base_pred_0 = [0]*83
tfidf_pred_0 = [1]*81 + [0]*2  # [[12  2] over [ 5 81]], was [[11  3] over [ 5 81]]
USE_sk_pred_0 = [1]*80 + [0]*3  # [[11  3] over [ 6 80]], was [[10  3] over [ 6 81]]
sBERT_sk_pred_0 = [1]*75 + [0]*8  # [[ 12  8] over [ 5 75]], was [[ 9  7] over [ 7 77]]
USE_keras_pred_0 = [1]*73 + [0]*10  # [[14  10] over [ 3 73]], was [[11  7] over [ 5 77]]
sBERT_keras_pred_0 = [1]*71 + [0]*12  # [[ 13  12] over [ 4 71]], was [[ 7  0] over [ 9 84]]  # all values updated
class_0_df = pd.DataFrame({'actual_0': actual_0, 'base_pred_0': base_pred_0,
                           'tfidf_pred_0': tfidf_pred_0, 'USE_sk_pred_0': USE_sk_pred_0,
                           'sBERT_sk_pred_0': sBERT_sk_pred_0, 'USE_keras_pred_0': USE_keras_pred_0,
                           'sBERT_keras_pred_0': sBERT_keras_pred_0})

models_list = ['actual', 'base_pred', 'tfidf_pred', 'USE_sk_pred',
                 'sBERT_sk_pred', 'USE_keras_pred', 'sBERT_keras_pred']

for ml in models_list:  # print out all accuracy and f1
    vari_1 = ml + "_1"
    vari_0 = ml + "_0"
    print(ml + " - macro_f1: " + str(macro_f1(class_1_df[vari_1], class_0_df[vari_0])))
    print(ml + " - accuracy: " + str(sum(class_1_df[vari_1]) + sum(class_0_df[vari_0])))

class_0_df.columns = class_1_df.columns
all_preds = class_1_df.append(class_0_df, ignore_index=True)

z = np.array([94,197,16,38,99,141,23])
y = np.array([52,104,146,10,51,30,40,27,46])
z = np.array(all_preds['base_pred_1'])
y = np.array(all_preds['tfidf_pred_1'])

theta_hat = z.mean() - y.mean()
# make array all predictions for each model

def run_permutation_test(pooled, sizeZ, sizeY, delta):
     np.random.shuffle(pooled)
     starZ = pooled[:sizeZ]
     starY = pooled[-sizeY:]
     return starZ.mean() - starY.mean()

pooled = np.hstack([z,y])
delta = z.mean() - y.mean()
numSamples = 10000
estimates = np.array(list(map(lambda x: run_permutation_test(pooled,z.size,y.size,delta),range(numSamples))))
diffCount = len(np.where(estimates <= delta)[0])
hat_asl_perm = 1.0 - (float(diffCount)/float(numSamples))
print(hat_asl_perm)

# # # #
test_stat_list = []
ap = all_preds[['base_pred_1', 'tfidf_pred_1']]
for i in range(10):
    new_list = []
    for row in range(len(ap)):
        new_list.append(ap.loc[row, ].sample(frac=1).values)
    df_permu = pd.DataFrame(new_list)
    test_stat = macro_f1(df_permu.iloc[0, :16], df_permu.iloc[0, 17:]) - macro_f1(df_permu.iloc[1, :16], df_permu.iloc[1, 17:])
    # restart above line - the 16 isn't gettring the top 16 rows...
    test_stat_list.append(test_stat)

def permute(n_permutes, worse_pred_1, worse_pred_0, better_pred_1, better_pred_0):
    test_stat_list = []
    all_pred_1 = pd.Series(list(worse_pred_1) + list(better_pred_1))
    all_pred_0 = pd.Series(list(worse_pred_0) + list(better_pred_0))
    for i in range(n_permutes):
        random.shuffle(all_pred_1)
        random.shuffle(all_pred_0)
        test_stat = macro_f1(all_pred_1[:16], all_pred_0[:100]) - macro_f1(all_pred_1[16:], all_pred_0[100:])
        test_stat_list.append(test_stat)
    return test_stat_list

permus = 10000  # 10000 on http://www2.stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html

og_test_stat = macro_f1(class_1_df['tfidf_pred_1'], class_0_df['tfidf_pred_0']) - \
               macro_f1(class_1_df['base_pred_1'], class_0_df['base_pred_0'])
pmtts = permute(permus, class_1_df['base_pred_1'], class_0_df['base_pred_0'], class_1_df['tfidf_pred_1'], class_0_df['tfidf_pred_0'])
diffCount = len(np.where(pmtts <= og_test_stat)[0])
print(diffCount)
hat_asl_perm = 1.0 - (float(diffCount)/float(permus))  # http://www2.stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html
print(hat_asl_perm < 0.05)

# you create a loop where 'worse pred' vals change for the diff embeddings predictions
# this prints out whether each of the worse preds are signif worse

models_list = ['USE_sk_pred_', 'sBERT_sk_pred_', 'USE_keras_pred_', 'sBERT_keras_pred_']  # different (smaller) list

for ml in models_list:
    print(ml)
    og_test_stat = macro_f1(class_1_df['tfidf_pred_1'], class_0_df['tfidf_pred_0']) - \
                   macro_f1(class_1_df[ml + str(1)], class_0_df[ml + str(0)])
    pmtts = permute(permus, class_1_df[ml + str(1)], class_0_df[ml + str(0)], class_1_df['tfidf_pred_1'],
                    class_0_df['tfidf_pred_0'])
    diffCount = len(np.where(pmtts <= og_test_stat)[0])
    print(diffCount)
    hat_asl_perm = 1.0 - (float(diffCount) / float(permus))
    print(hat_asl_perm < 0.05)




#_____________________________________________________________________________________________________________________

# write a loop that scrapes, embeds and classifies all non-labelled cases
import time

df2 = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled_h.csv")  # '_h' added to fix heathcote (although has no effect as only non-labelled used)
df2 = df2[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
df2 = df2[df2['Sample'] == 0]  # retain non-labelled judgments only (1137)
#df2 = df2.head(3) # for trialling

link_dict2 = df2.set_index('Link').T.to_dict('list')  # make a dict with Link as key

d = []
for l in enumerate(link_dict2.keys()):

    req = Request(l[1], headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html5lib")
    case_text = page_soup.get_text()  # scrape no more specif due to varying page layouts
    case_text = re.sub('\n', ' ', case_text)  # replace '\n' with ' '
    case_text = re.sub('@media screen|@media print|#screenonly|BAILII|Multidatabase Search|World Law', '', case_text)  # remove some patterns

    case_text = str(case_text)
    case_text = remove_urls(case_text)
    case_text = remove_html_tags(case_text)
    case_text = case_text.lower()
    case_text = case_text.strip()
    case_text = re.sub(r'\d+', '', case_text)
    case_text = remove_punct(case_text)
    case_text = remove_first_n_words(case_text)
    case_text = retain_english(case_text)

    # add in some sleep to script
    ran = np.random.random_integers(0, 5)
    time.sleep(1 + ran)

    # and now you need to apply ur final tfidf model to the case text (which also tf-idf transforms it)
    d.append(
        {
            'link': l[1],
            'case_name': link_dict2[l[1]][0],
            'year': link_dict2[l[1]][1],
            'ml_classification': final_model.predict([case_text])[0],
            'word_count_once_manipd': len(word_tokenize(case_text)),
            'manipd_text': case_text  # line should be removed when uploading to github - you say not stored
        }
    )

    print("Done " + str(l[0]+1) + " classifications")

print("done")

non_labelled_d = pd.DataFrame(d)

# DATA SAVING
#non_labelled_d.to_csv('/Users/joewatson/Desktop/LawTech/predicted_non_labelled_cases.csv', index=False)
# DATA LOADING
#pd.read_csv('/Users/joewatson/Desktop/LawTech/predicted_non_labelled_cases.csv')

#non_l_d_c = non_labelled_d[['case_name', 'year', 'link', 'my_classification', 'raw_classification']].copy()  # retain cols to match with imported df
non_l_d_c = non_labelled_d[['case_name', 'year', 'link', 'ml_classification']].copy()  # retain cols to match with imported df
non_l_d_c['sample'] = 0  # adding sample column for upcoming concat
non_l_d_c['sme_classification'] = np.nan  # adding march_classification column for upcoming concat
non_l_d_c['sme_narrow'] = np.nan  # adding march_narrow column for upcoming concat
non_l_d_c['class_match'] = np.nan

# # # # # # # # # #

# make a df with my_classification and march_classification columns

# use df created at start
df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled_h.csv")  # import same csv with Heathcote error fixed
df = df[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
df = df[df['Classification'] >= 0]  # retain labelled judgments only
df['sme_narrow'] = np.where(df['Classification'] == 1, 1, 0)  # make a narrow class column, covering animal protection law only
df.columns = ['case_name', 'year', 'link', 'sme_classification', 'sample', 'sme_narrow']  # renaming the cols of the df created early on, which holds only labelled judgments

judgments_preds = pd.concat([pd.DataFrame(y_pred).reset_index(drop=True),
                             pd.DataFrame(X_test['Link']).reset_index(drop=True)], axis=1)
# link changed for Link in above line, to facil use of data from data loading point
judgments_preds.columns = ['ml_classification', 'link']

labelled_n_preds = pd.merge(df, judgments_preds, how='left')  # merge df with judgments_preds based on 'link'

conditions = [
    (labelled_n_preds['ml_classification'] == 0) & (labelled_n_preds['sme_narrow'] == 0),
    (labelled_n_preds['ml_classification'] == 1) & (labelled_n_preds['sme_narrow'] == 1),
    (labelled_n_preds['ml_classification'] == 0) & (labelled_n_preds['sme_narrow'] == 1),
    (labelled_n_preds['ml_classification'] == 1) & (labelled_n_preds['sme_narrow'] == 0),
    (labelled_n_preds['ml_classification'].isnull())
    ]  # https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/

# create a list of the values we want to assign for each condition
values = [1, 1, 0, 0, np.nan]

# create a new column and use np.select to assign values to it using our lists as arguments
labelled_n_preds['class_match'] = np.select(conditions, values)

#concat (stack) with non_l_d_c and write to csv
full_pred_df = pd.concat([non_l_d_c, labelled_n_preds])
full_pred_df.columns = ['case_name', 'year', 'link', 'ml_class', 'sme_sample',
                        'sme_class', 'sme_narrow_class', 'class_match']  # rename columns

# writing a df that shows where classifications differ
#full_pred_df.to_csv("/Users/joewatson/Desktop/LawTech/full_pred_df27_apr.csv", index=False)  # just filter class_match == 0

a4a_df = full_pred_df.copy()
a4a_df = a4a_df[['case_name', 'year', 'link', 'ml_class', 'sme_narrow_class']]

classification_list = []
for i in range(len(a4a_df)):
    classifications = []
    if a4a_df.iloc[i, 4] >= 0:  # march_narrow classification available
        classifications = [a4a_df.iloc[i, 4], 'expert']
    else:
        classifications = [a4a_df.iloc[i, 3], 'ml']
    classification_list.append(classifications)

# concat SME's narrow classification - or ml classification if SME's unavailable - to first 3 a4a_df cols
a4a_df_share = pd.concat([a4a_df[['case_name', 'year', 'link']].reset_index(drop=True), pd.DataFrame(classification_list)], axis=1)
a4a_df_share.columns = ['case_name', 'year', 'link', 'classification', 'expert_or_ml_classified']
a4a_df_share['classification'] = np.where(a4a_df_share['link'] == 'https://www.bailii.org/ew/cases/EWHC/Admin/2002/908.html', 1, a4a_df_share['classification'])  # fix labelling error
a4a_df_share = a4a_df_share.sort_values('year')
#a4a_df_share.to_csv("/Users/joewatson/Desktop/LawTech/a4a_df_share4May.csv", index=False)
