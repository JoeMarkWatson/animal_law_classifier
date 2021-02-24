import pandas as pd
import numpy as np
import tensorflow_hub as hub
import re
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from urllib.request import Request
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)  # https://stackoverflow.com/questions/58638701/importerror-cannot-import-name-set-random-seed-from-tensorflow-c-users-po
import random
from sklearn.model_selection import train_test_split


# # # judgment gathering # # #

# import labelled spreadsheet
# df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled.csv")  # import csv with March's labels on it
df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled_h.csv")  # csv with Heathcote error fixed
df = df[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
df = df[df['Classification'] >= 0]  # retain labelled judgments only

# make a df with Link as key
link_dict = df.set_index('Link').T.to_dict('list')
# demo how the list works
print(link_dict['https://www.bailii.org/uk/cases/UKPC/2000/47.html'][0])  # case_name
print(link_dict['https://www.bailii.org/uk/cases/UKPC/2000/47.html'][1])  # year
print(link_dict['https://www.bailii.org/uk/cases/UKPC/2000/47.html'][2])  # classification

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
            'classification': link_dict[l][2],
            'word_count_pre_stem': len(word_tokenize(case_text)),  # included here to help catch scrape error
            'judgment_text': case_text
        }
    )
print("done")

dd = pd.DataFrame(d)

dd['classification_narrow'] = np.where(dd['classification'] == 1, 1, 0)
dd['classification_broad'] = np.where(dd['classification'] > 0, 1, 0)

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

dd['jtfc'] = dd['judgment_text'].map(str) \
                            .map(lambda x: x.lower()) \
                            .map(lambda x: x.strip()) \
                            .map(lambda x: re.sub(r'\d+', '', x)) \
                            .map(remove_punct) \
                            .map(remove_urls) \
                            .map(remove_html_tags)


# create y and X (with two versions of X for use in embeddings-based and word freq-based models)
X, y = dd[['judgment_text', 'jtfc']], dd['classification_narrow']

# # # creating a tfidf model # # #

#########¡¡¡¡¡¡€€€€€€€€###### RESTART WITH THE BELOW - YET TO TRIAL ANYTHING AND PACKAGES STILL PROB NEED IMPORTING
# https://medium.com/@am.benatmane/keras-hyperparameter-tuning-using-sklearn-pipelines-grid-search-with-cross-validation-ccfc74b0ce9f

from keras.layers import Dense, Input, Dropout

def create_model(optimizer="adam", dropout=0.1, init='uniform', nbr_features=2500, dense_nparams=256):
    model = Sequential()
    model.add(Dense(dense_nparams, activation='relu', input_shape=(nbr_features,), kernel_initializer=init,))
    model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model

keras_estimator = KerasClassifier(build_fn=create_model, verbose=1)

estimator = Pipeline([("tfidf", TfidfVectorizer(analyzer ="word",
                                                max_features=2500,)),
                       ('ss', StandardScaler(with_mean=False,)),
                       ("kc", kears_estimator)])

param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2), (2,2), (1,3)],
    'tfidf__use_idf': [True, False],
    'kc__epochs': [10, 100, ],
    'kc__dense_nparams': [32, 256, 512],
    'kc__init': [ 'uniform', 'zeros', 'normal', ],
    'kc__batch_size':[2, 16, 32],
    'kc__optimizer':['RMSprop', 'Adam', 'Adamax', 'sgd'],
    'kc__dropout': [0.5, 0.4, 0.3, 0.2, 0.1, 0]
}

X = data.review
y = data.sentiment
kfold_splits = 5
grid = GridSearchCV(estimator=estimator,
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=True,
                    cv=kfold_splits,  #StratifiedKFold(n_splits=kfold_splits, shuffle=True)
                    param_grid=param_grid,)

grid_result = grid.fit(X, y, ) #callbacks=[tbCallBack]

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# # # embedding # # #

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
    wt_list_vals = [wt for wt in wt_list_vals if len(wt) < 2000]
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

print(all_mean_embs.shape)

# DATA SAVING POINT BELOW
#all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')  # written 08.01.2021
#y.to_csv('/Users/joewatson/Desktop/LawTech/labelled_labels8Jan.csv')  # written 08.01.2021
# DATA SAVING POINT ABOVE

# DATA LOADING POINT BELOW
#all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')
#all_mean_embs = all_mean_embs.iloc[:, 1:]
#y = pd.read_csv('/Users/joewatson/Desktop/LawTech/labelled_labels8Jan.csv')
#y = y.iloc[:, 1:]
#df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled.csv")
#df = df[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
#df = df[df['Classification'] >= 0]  # retain labelled judgments only
#link_dict = df.set_index('Link').T.to_dict('list')
# DATA LOADING POINT ABOVE

# # # MLing # # #

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold  # likely more applicable than Kfold as unbalanced classes: http://ethen8181.github.io/machine-learning/model_selection/model_selection.html#K-Fold-Cross-Validation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def create_model(learning_rate, activation, shape_number_a):  # then add batch size later, leaving no. of layers and layer order constant
    opt = Adam(lr=learning_rate)  # create an Adam optimizer with the given learning rate

    model = Sequential()

    model.add(Dense(shape_number_a, input_shape=(512,), activation=activation))  # create input layer
    # this is a good way of doing this if you want to trial multiple layers: https://datagraphi.com/blog/post/2019/12/17/how-to-find-the-optimum-number-of-hidden-layers-and-nodes-in-a-neural-network-model
    model.add(Dense(1, activation='sigmoid'))  # create output layer

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model with optimizer, loss, and metrics

    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu'], 'batch_size': [1, 5, 10],
          'epochs': [20, 50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001], 'shape_number_a': [256, 128, 64, 32]}
         # other optimisers available, with adam by far most common  # https://keras.io/api/optimizers/

np.random.seed(1)
seed(1)  # from numpy again I think so poss duplicating
tensorflow.random.set_seed(1)  # from tf

all_mean_embs = pd.concat([all_mean_embs, df['Link'].reset_index(drop=True)], axis=1)  # adding link onto end of all_mean_embs for later merging
# in the above line, dd['link'] was changed to df['Link'].reset_index(drop=True) for quicker loading (from above
# data loading point)

X_train, X_test, y_train, y_test = train_test_split(all_mean_embs, y, test_size=0.2, random_state=1)  # 0.25 is default
# but 0.2 gives 100 test samples (so appears logical for reporting purposes)

random_search = RandomizedSearchCV(model, param_distributions=params, cv=StratifiedKFold(5), n_jobs=-1, random_state=1)  # cannot be fully reproducible as not single threaded: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development  https://datascience.stackexchange.com/questions/37413/why-running-the-same-code-on-the-same-data-gives-a-different-result-every-time

#******** do not run below lines when quick loading as takes 5 mins searching
ran_result = random_search.fit(X_train.iloc[:, :-1], y_train)  # takes 5 mins
print("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

# Best accuracy: 0.9 (precisely)  # accuracy in write up
# Best combination: {'shape_number_a': 256, 'learning_rate': 0.1, 'epochs': 20, 'batch_size': 5, 'activation': 'relu'}  # model in write up
# When model identified without stratifying kfolds  # Best accuracy: 0.8975000023841858
# When model identified without stratifying kfolds  # Best combination: {'shape_number_a': 64, 'learning_rate': 0.001, 'hidden_layers': 1, 'epochs': 50, 'batch_size': 10, 'activation': 'relu'}

ran_result.best_estimator_.model.save("/Users/joewatson/Desktop/LawTech/ran_search_strat_model.hdf5")  # save the best model
#ran_result.best_estimator_.model.save("/Users/joewatson/Desktop/LawTech/ran_search_model.hdf5")  # when model identified without stratifying kfolds
#******** do not run above lines when quick loading as takes 5 mins searching

best_model = tensorflow.keras.models.load_model("/Users/joewatson/Desktop/LawTech/ran_search_strat_model.hdf5")  # https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model  # compile=True appears not to be required
#y_pred_proba = best_model.predict_proba(X_test.iloc[:, :-1])  # in case prediction info is required  # likely deprecated method but works consistently
y_pred_proba = best_model.predict(X_test.iloc[:, :-1])  # in case prediction info is required  # code altered to avoid deprecation warning
y_pred = (best_model.predict(X_test.iloc[:, :-1]) > 0.5).astype("int32")

print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
print(classification_report(y_pred, y_test))
print(accuracy_score(y_pred, y_test))

judgments_preds = pd.concat([pd.DataFrame(y_pred).reset_index(drop=True),
                             pd.DataFrame(y_pred_proba).reset_index(drop=True),
                             pd.DataFrame(X_test['Link']).reset_index(drop=True)], axis=1)
# link changed for Link in above line, to facil use of data from data loading point
judgments_preds.columns = ['my_classification', 'raw_classification', 'link']


# # # # # # # # # #

# change 'animal' to 'dog', 'horse', 'cat' to show that classifier could still function without the presence of 'animal'

animal_list = ['dog', 'horse', 'cat']
d = []

for a in animal_list:

    for l in enumerate(judgments_preds['link']):

        req = Request(l[1], headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        page_soup = BeautifulSoup(webpage, "html5lib")
        case_text = page_soup.get_text()  # scrape no more specif due to varying page layouts
        case_text = re.sub('\n', ' ', case_text)  # replace '\n' with ' '
        case_text = re.sub('@media screen|@media print|#screenonly|BAILII|Multidatabase Search|World Law', '', case_text)  # remove some patterns
        case_text = re.sub('\\banimals\\b', str(a + "s"), case_text)  # https://stackoverflow.com/questions/3995034/do-regular-expressions-from-the-re-module-support-word-boundaries-b
        case_text = re.sub('\\banimal\\b', a, case_text)

        wt_list = []
        wt_list.append(sent_tokenize(case_text))
        wt_list_vals = pd.DataFrame(wt_list).values
        wt_list_vals = wt_list_vals.flatten()
        print("Original judgment sentence count is " + str(len(wt_list_vals)))
        wt_list_vals = [wt for wt in wt_list_vals if len(wt) < 2000]
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

        d.append(
            {
                'sub_word': a,
                'link': l[1],
                'case_name': link_dict[l[1]][0],
                'year': link_dict[l[1]][1],
                'word_count_pre_stem': len(word_tokenize(case_text)),
                'sent_count_pre_stem': len(sent_tokenize(case_text)),
                'my_classification': (best_model.predict(means_df) > 0.5).astype("int32")[0][0],
                'raw_classification': best_model.predict(means_df)[0][0]  # which can be used for certainty
            }
        )

        print("Done " + str(l[0]+1) + " classifications")

print("done")

dhc = pd.DataFrame(d)

#SAVING AND LOADING POINT BELOW
#dhc.to_csv("/Users/joewatson/Desktop/LawTech/dhc_21Jan.csv", index=False)  # saved in case want to re-inspect
#judgments_preds.to_csv("/Users/joewatson/Desktop/LawTech/judgment_preds_test_set_21Jan.csv", index=False)  # saved in case want to re-inspect

#dhc = pd.read_csv("/Users/joewatson/Desktop/LawTech/dhc_21Jan.csv")
#judgments_preds = pd.read_csv("/Users/joewatson/Desktop/LawTech/judgment_preds_test_set_21Jan.csv")
#SAVING AND LOADING POINT ABOVE

for a in animal_list:
    for row in range(len(judgments_preds)):
        if dhc[dhc['sub_word'] == a][['my_classification']].reset_index(drop=True).loc[row][0] != judgments_preds['my_classification'][row]:
            print('When sub word is ' + a + ', predictions differ for: ')
            print(dhc[dhc['sub_word'] == a][['link']].reset_index(drop=True).loc[row][0])
            # with no printout showing that no predictions differ from original predictions

# When sub word is dog, predictions differ for:
# https://www.bailii.org/ew/cases/EWCA/Crim/2012/1288.html
# When sub word is dog, predictions differ for:
# https://www.bailii.org/ew/cases/EWCA/Civ/2006/632.html
# When sub word is horse, predictions differ for:
# https://www.bailii.org/ew/cases/EWHC/Admin/2010/347.html

# # # # # # # # # #

# write a loop that scrapes, embeds and classifies all non-labelled cases

df2 = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled.csv")
df2 = df2[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
df2 = df2[df2['Sample'] == 0]  # retain non-labelled judgments only (1137)

link_dict2 = df2.set_index('Link').T.to_dict('list')  # make a dict with Link as key

d = []
for l in enumerate(link_dict2.keys()):

    req = Request(l[1], headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html5lib")
    case_text = page_soup.get_text()  # scrape no more specif due to varying page layouts
    case_text = re.sub('\n', ' ', case_text)  # replace '\n' with ' '
    case_text = re.sub('@media screen|@media print|#screenonly|BAILII|Multidatabase Search|World Law', '', case_text)  # remove some patterns

    wt_list = []
    wt_list.append(sent_tokenize(case_text))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()
    print("Original judgment sentence count is " + str(len(wt_list_vals)))
    wt_list_vals = [wt for wt in wt_list_vals if len(wt) < 2000]
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

    d.append(
        {
            'link': l[1],
            'case_name': link_dict2[l[1]][0],
            'year': link_dict2[l[1]][1],
            'word_count_pre_stem': len(word_tokenize(case_text)),
            'sent_count_pre_stem': len(sent_tokenize(case_text)),
            'my_classification': (best_model.predict(means_df) > 0.5).astype("int32")[0][0],
            'raw_classification': best_model.predict(means_df)[0][0]  # which can be used for certainty
        }
    )

    print("Done " + str(l[0]+1) + " classifications")

print("done")

non_labelled_d = pd.DataFrame(d)
#non_labelled_d['my_classification'] = list(map(lambda x: x[0][0], non_labelled_d['my_classification']))
non_l_d_c = non_labelled_d[['case_name', 'year', 'link', 'my_classification', 'raw_classification']].copy()  # retain cols to match with imported df
non_l_d_c['sample'] = 0  # adding sample column for upcoming concat
non_l_d_c['march_classification'] = np.nan  # adding march_classification column for upcoming concat
non_l_d_c['march_narrow'] = np.nan  # adding march_narrow column for upcoming concat
non_l_d_c['class_match'] = np.nan


# # # # # # # # # #

# make a df with my_classification and march_classification columns

# use df created at start
#df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled.csv")  # import csv with March's labels on it
#df = df[['Case', 'Year', 'Link', 'Classification', 'Sample']]  # remove Index, Explanation and og_sample columns
#df = df[df['Classification'] >= 0]  # retain labelled judgments only
df.columns = ['case_name', 'year', 'link', 'march_classification', 'sample']  # renaming the cols of the df created early on, which holds only labelled judgments
df['march_narrow'] = np.where(df['march_classification'] == 1, 1, 0)

labelled_n_preds = pd.merge(df, judgments_preds, how='left')  # merge df with judgments_preds based on 'link'

conditions = [
    (labelled_n_preds['my_classification'] == 0) & (labelled_n_preds['march_narrow'] == 0),
    (labelled_n_preds['my_classification'] == 1) & (labelled_n_preds['march_narrow'] == 1),
    (labelled_n_preds['my_classification'] == 0) & (labelled_n_preds['march_narrow'] == 1),
    (labelled_n_preds['my_classification'] == 1) & (labelled_n_preds['march_narrow'] == 0),
    (labelled_n_preds['my_classification'].isnull())
    ]  # https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/

# create a list of the values we want to assign for each condition
values = [1, 1, 0, 0, np.nan]

# create a new column and use np.select to assign values to it using our lists as arguments
labelled_n_preds['class_match'] = np.select(conditions, values)

#concat (stack) with non_l_d_c and write to csv
full_pred_df = pd.concat([non_l_d_c, labelled_n_preds])

# writing to csv so March can see where human and machine classifications are different
#full_pred_df.to_csv("/Users/joewatson/Desktop/LawTech/full_pred_df21_jan.csv", index=False)

# # # # #

# writing csv with human and machine labelled judgments for A4A to use as internal resource
a4a_df = pd.read_csv("/Users/joewatson/Desktop/LawTech/full_pred_df21_jan.csv")
a4a_df = a4a_df[['case_name', 'year', 'link', 'my_classification', 'raw_classification', 'march_narrow']]

classification_list = []
for i in range(len(a4a_df)):
    player_classification = []
    if a4a_df.iloc[i, 5] >= 0:  # march_narrow classification available
        player_classification = [a4a_df.iloc[i, 5], np.nan]
    else:
        player_classification = [a4a_df.iloc[i, 3], np.round(a4a_df.iloc[i, 4], 4)]
    classification_list.append(player_classification)

# concat march's narrow classification - or your classification if march's unavailable - to first 3 a4a_df cols
a4a_df_share = pd.concat([a4a_df[['case_name', 'year', 'link']], pd.DataFrame(classification_list)], axis=1)
a4a_df_share.columns = ['case_name', 'year', 'link', 'classification', 'raw_pred_if_non-human']
a4a_df_share = a4a_df_share.sort_values('year')
#a4a_df_share.to_csv("/Users/joewatson/Desktop/LawTech/a4a_df_share21Jan.csv", index=False)







# re no. of param.s vs no. of training samples: https://stats.stackexchange.com/questions/329861/what-happens-when-a-model-is-having-more-parameters-than-training-samples

# re regularisation: assumed not needed for now. Incog tab link: https://towardsdatascience.com/regularization-techniques-and-their-implementation-in-tensorflow-keras-c06e7551e709
# see also: https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/

# re activations: # https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
# good general article: https://towardsdatascience.com/are-you-using-the-scikit-learn-wrapper-in-your-keras-deep-learning-model-a3005696ff38

# https://towardsdatascience.com/classifying-scientific-papers-with-universal-sentence-embeddings-4e0695b70c44
# https://towardsdatascience.com/comparison-of-dimensionality-reduction-techniques-for-image-art-style-embeddings-723912fd6250
# https://www.topbots.com/document-embedding-techniques/#choose-technique
# https://towardsdatascience.com/using-use-universal-sentence-encoder-to-detect-fake-news-dfc02dc32ae9
