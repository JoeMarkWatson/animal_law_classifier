from numpy.random import seed
seed(1)
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import tensorflow
tensorflow.random.set_seed(1)  # https://stackoverflow.com/questions/58638701/importerror-cannot-import-name-set-random-seed-from-tensorflow-c-users-po
max_f = 768
from keras.constraints import maxnorm


def create_keras_mlp(learning_rate, activation, dense_nparams, dropout):  # leaving no. of layers and layer order constant
    opt = Adam(learning_rate=learning_rate)  # create an Adam optimizer with the given learning rate
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(max_f,)))  # https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model.add(Dense(dense_nparams, activation=activation, kernel_constraint=maxnorm(dropout*15)))  # create input layer
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))  # create output layer
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model with optimizer, loss, and metrics  # to maybe get F1: https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
    return model


def mlp_approach(X_train, y_train, X_test, y_test):
    param_grid = {
        'epochs': [10, 20, 50, 100],
        'dense_nparams': [max_f / 16, max_f / 8, max_f / 4, max_f / 2],
        'learning_rate': [0.1, 0.01, 0.001],
        'activation': ['relu'],
        'dropout': [0.2, 0]
    }

    model = KerasClassifier(build_fn=create_keras_mlp)

    np.random.seed(1)
    seed(1)
    tensorflow.random.set_seed(1)
    random_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(5), n_jobs=1,
                                 scoring='f1_macro', refit='f1_macro')
    ran_result = random_search.fit(X_train, y_train, verbose=0)  # takes 5-10 minutes
    print("Best f1_macro: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "./_longformer_mlp_score_params.txt"
    with open(loc_string, "w") as text_file:
        text_file.write("Best accuracy: {}\nBest combination: {}".format(ran_result.best_score_, ran_result.best_params_))

    loc_string = "./new_ran_search_strat_model_longformer_mlp.hdf5"
    ran_result.best_estimator_.model.save(loc_string)

    best_model = tensorflow.keras.models.load_model(loc_string)

    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")

    print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
    print(classification_report(y_pred, y_test))
    print("f1", f1_score(y_pred, y_test, average="macro"))
    print(accuracy_score(y_pred, y_test))


def svm_approach(X_train, y_train, X_test, y_test):
    svm = LinearSVC(max_iter=10000)  # increasing max_iter (suggested by convergence fail error message when running sBERT,
    # and one of the multiple poss options given here: https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    params = {'C': (0.1, 1, 2, 5, 10),
            'loss': ('hinge', 'squared_hinge')}  # gamma not applicable to LinearSVC

    grid_search = GridSearchCV(svm, params, cv=StratifiedKFold(5), n_jobs=-1, verbose=1, scoring='f1_macro', refit='f1_macro')
    grid_result = grid_search.fit(X_train, y_train)
    print("Best f1_macro: {}\nBest combination: {}".format(grid_result.best_score_, grid_result.best_params_))
    # save model and best params
    loc_string = "./sk_score_params_Longformer_svm_hinge.txt"  # for saving hinge only
    with open(loc_string, "w") as text_file:
        text_file.write("Best macro_f1: {}\nBest combination: {}".format(grid_result.best_score_, grid_result.best_params_))
    loc_string = "./new_search_sk_Longformer_svm_hinge.hdf5"  # for saving hinge only
    dump(grid_search.best_estimator_, loc_string)
    # load model
    # implement on test set
    y_pred = (grid_result.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
    print("f1", f1_score(y_pred, y_test, average="macro"))
    print(accuracy_score(y_pred, y_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['longformer','sbert','bigbird'], required=True)
    parser.add_argument('--clf', choices=['svm','mlp'], required=True)
    args = parser.parse_args()
    
    # load files produced by transformer_embedding_extraction.py
    embeds = np.load(f'{args.model}_embeds.npy')
    with open(f'{args.model}_urls.txt') as f:
        urls = f.read().split("\n")

    with open('train.csv') as f:
        X_train, y_train = zip(*[a.split(",") for a in f.read().split("\n")])
        X_train = list(X_train)
        y_train = [int(a) for a in y_train]
    with open('test.csv') as f:
        X_test, y_test = zip(*[a.split(",") for a in f.read().split("\n")])
        X_test = list(X_test)
        y_test = [int(a) for a in y_test]

    X_train_i = [urls.index(a) for a in X_train]
    X_test_i = [urls.index(a) for a in X_test]

    X_train = embeds[X_train_i]
    X_test = embeds[X_test_i]

    if args.clf == 'mlp':
        mlp_approach(X_train, y_train, X_test, y_test)
    elif args.clf == 'svm':
        svm_approach(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()

