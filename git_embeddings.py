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
import pandas
from numpy.random import seed
seed(1)
import tensorflow
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(1)
from numpy.random import seed
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# # # judgment gathering # # #

# import labelled spreadsheet
df = pd.read_csv("/Users/joewatson/Desktop/LawTech/animal_df2_labelled.csv")
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

# the below was applied to fix error from calling link_dict wrong in (now fixed) loop above
#dd.columns = ['link', 'year', 'classification', 'case_name', 'word_count_pre_stem', 'judgment_text']
#dd['case_name'] = list(map(lambda x: link_dict[x][0], dd['link']))
# the above was applied to fix error from calling link_dict wrong in (now fixed) loop above


# # # embedding # # #

# load USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
embed = hub.load(module_url)

dd['classification_narrow'] = np.where(dd['classification'] == 1, 1, 0)
dd['classification_broad'] = np.where(dd['classification'] > 0, 1, 0)
X, y = dd['judgment_text'], dd['classification_narrow']

all_mean_embs = pd.DataFrame()
for jt in dd['judgment_text']:
    wt_list = []
    wt_list.append(sent_tokenize(jt))
    wt_list_vals = pd.DataFrame(wt_list).values
    wt_list_vals = wt_list_vals.flatten()  # works
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

#all_mean_embs.to_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')  # written 08.01.2021
#y.to_csv('/Users/joewatson/Desktop/LawTech/labelled_labels8Jan.csv')  # written 08.01.2021

#all_mean_embs = pd.read_csv('/Users/joewatson/Desktop/LawTech/labelled_embeddings8Jan.csv')
#all_mean_embs = all_mean_embs.iloc[:,1:]

# # # MLing # # #

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def create_model(learning_rate, activation, shape_number_a, shape_number_b, hidden_layers):  # then add batch size later, leaving no. of layers and layer order constant
    opt = Adam(lr=learning_rate)  # create an Adam optimizer with the given learning rate

    model = Sequential()
    model.add(Dense(shape_number_a, input_shape=(512,), activation=activation))  # create input layer

    for i in range(hidden_layers):
        if shape_number_a >= shape_number_b:
            model.add(Dense(shape_number_b, activation=activation))

    model.add(Dense(1, activation='sigmoid'))  # create output layer

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model with optimizer, loss, and metrics

    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [1, 5, 10],
          'epochs': [20, 50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001], 'shape_number_a': [256, 128, 64, 32],
          'shape_number_b': [256, 128, 64, 32], 'hidden_layers': [1, 2]}  # other optimisers available, with adam by far most common  # https://keras.io/api/optimizers/

# Create a randomize search cv object passing in the parameters to try
seed(1)  # from numpy
tensorflow.random.set_seed(1)  # from tf, but might not be reqd

all_mean_embs = pd.concat([all_mean_embs, dd['link']], axis=1)  # adding link onto end of all_mean_embs for later merging

X_train, X_test, y_train, y_test = train_test_split(all_mean_embs, y, test_size=0.2, random_state=1)  # 0.25 is default
# but 0.2 gives 100 test samples (so appears logical for reporting purposes)
random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(5), n_jobs=-1)
ran_result = random_search.fit(X_train.iloc[:, :-1], y_train)  # takes > 20 mins (started 13:35, printing 'now' with time)

print("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))
# Best accuracy: 0.8975000023841858
# Best combination: {'shape_number_b': 128, 'shape_number_a': 64, 'learning_rate': 0.001, 'hidden_layers': 1, 'epochs': 50, 'batch_size': 10, 'activation': 'relu'}

best_model = ran_result.best_estimator_  # this seems to be the way of keeping best estimator found: https://www.kaggle.com/arrogantlymodest/randomised-cv-search-over-keras-neural-network
y_pred_proba = best_model.predict_proba(X_test.iloc[:, :-1])
y_pred = best_model.predict(X_test.iloc[:, :-1])

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
print(classification_report(y_pred, y_test))
print(accuracy_score(y_pred, y_test))

judgments_preds = pd.concat([pd.DataFrame(y_pred).reset_index(drop=True), pd.DataFrame(X_test['link']).reset_index(drop=True)], axis=1)
judgments_preds.columns = ['my_classification', 'link']

# # # # # # # # # #

# write a loop that scrapes, embeds and classifies

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
            'my_classification': best_model.predict(means_df)
        }
    )

    print("Done " + str(l[0]+1) + " classifications")

print("done")

non_labelled_d = pd.DataFrame(d)
non_labelled_d['my_classification'] = list(map(lambda x: x[0][0], non_labelled_d['my_classification']))
non_l_d_c = non_labelled_d[['case_name', 'year', 'link', 'my_classification']].copy()  # retain cols to match with imported df
non_l_d_c['sample'] = 0  # adding sample column for upcoming concat
non_l_d_c['march_classification'] = np.nan  # adding march_classification column for upcoming concat
non_l_d_c['march_narrow'] = np.nan  # adding march_classification column for upcoming concat
non_l_d_c['class_match'] = np.nan

# make a df with my_classification and march_classification columns
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
full_pred_df.to_csv("/Users/joewatson/Desktop/LawTech/full_pred_df12_jan.csv", index=False)




# re activations  # https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
# good general article: https://towardsdatascience.com/are-you-using-the-scikit-learn-wrapper-in-your-keras-deep-learning-model-a3005696ff38

#_______________________________________________________________________________________________________________________

# create a keras model (outside the sklearn wrapper) using params obtained through experimentation

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def plot_loss(loss,val_loss):
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def plot_accuracy(acc,val_acc):
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# run a keras model without any tuning beyond saving the best epoch (with param.s set through experimentation)
model_save = ModelCheckpoint('best_model.hdf5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=100)
seed(123)  # from numpy
tensorflow.random.set_seed(123)  # from tf, with both required for reproducible results: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
model = Sequential()
model.add(Dense(128, input_shape=(512,), activation='relu'))
model.add(Dense(1, input_shape=(128,), activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()
h_callback = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[model_save, early_stopping])
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])
plot_accuracy(h_callback.history['acc'], h_callback.history['val_acc'])
#keras.models.load_model('path/to/location')  # to load the model back
