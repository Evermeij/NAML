from functools import partial,reduce
import os


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator

from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import make_scorer

from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer

from scipy.stats import randint, expon, norm

import numpy as np

import re

from flask import json

import pickle

from visualisations.visualisation import PYTHON_PATH_CONFUSION_MATRIX , \
                                           PYTHON_PATH_WORDCLOUD, \
                                           PYTHON_PATH_PIES, \
                                           PYTHON_PATH_ROCS, \
                                           PYTHON_PATH_MODELS
from database.postgresDB import get_mail,loadEmailsTrainData
from machineLearning.config import  tokenize,\
                                    PYTHON_PATH_JSON_INFO_EMAIL_IMAGES,\
                                    PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES,\
                                    PYTHON_PATH_THRESHOLD,\
                                    PYTHON_FILENAME_THRESHOLD,\
                                    LOCALDIR

from machineLearning.base import BaseMLProject


class MLProject(BaseMLProject):
    LOCALDIR = LOCALDIR

    PYTHON_PATH_THRESHOLD = PYTHON_PATH_THRESHOLD
    PYTHON_FILENAME_THRESHOLD = PYTHON_FILENAME_THRESHOLD

    PYTHON_PATH_MODELS = PYTHON_PATH_MODELS

    PYTHON_PATH_JSON_INFO_EMAIL_IMAGES = PYTHON_PATH_JSON_INFO_EMAIL_IMAGES
    PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES = PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES

    MODEL_DICTIONARY = {'mnb': MultinomialNB(fit_prior=False), 'rf': RandomForestClassifier(n_estimators=50),
                        'etr': ExtraTreesClassifier(n_estimators=50)}
    STOPS = set(stopwords.words('dutch'))
    DUTCHSTEMS = stemmer = DutchStemmer()

    # def __init__(self,name_model):
    #     self.name_model = name_model
    class weightEst(BaseEstimator):
        def __init__(self, gridModel,w_0, w_1, thres):
            self.gridModel = gridModel
            self.w_0 = w_0
            self.w_1 = w_1
            self.thres = thres

        def fit(self, X, y):
            weight = self.w_0 * (y == 0) + self.w_1 * (y == 1)
            self.gridModel.fit(X, y, **{'classifier__sample_weight': weight})
            return self

        def predict(self, X):
            score = self.gridModel.predict_proba(X)
            ypred = (score[:, 0] < self.thres).astype(int)
            return ypred

        def predict_proba(self, X):
            score = self.gridModel.predict_proba(X)
            return score

        def get_params(self, deep=True):
            params = {'gridModel':self.gridModel,'w_0': self.w_0, 'w_1': self.w_1, 'thres': self.thres}
            return params

        def set_params(self, **params):
            # self.gridModel = params['gridModel']
            self.w_0 = params['w_0']
            self.w_1 = params['w_1']
            self.thres = params['thres']
            return self

    def get_estimator(self,model_name = 'mnb'):
        model = self.MODEL_DICTIONARY[model_name]
        estimator = Pipeline(steps=[

            ('vectorizer', TfidfVectorizer(input=u'content', encoding=u'latin1', decode_error=u'strict', strip_accents=None,
                                           lowercase=True,
                                           preprocessor=None,
                                           tokenizer=partial(tokenize, stop=self.STOPS, stemmer=self.DUTCHSTEMS),
                                           analyzer=u'word',  # stop_words=(stopwords.words('dutch')),
                                           ngram_range=(1, 3),  # max_df=0.9, min_df=0.005,
                                           max_features=10000, vocabulary=None, binary=False,
                                           norm=u'l1', use_idf=True, smooth_idf=True, sublinear_tf=False)),

            ('classifier', model)
            ]
            )
        return estimator

    def get_threshold_dic(self):
        return np.load(self.PYTHON_PATH_THRESHOLD + self.PYTHON_FILENAME_THRESHOLD).item()

    def set_threshold_dic(self,name_model, new_thres):
        old_thresholds = np.load(self.PYTHON_PATH_THRESHOLD + self.PYTHON_FILENAME_THRESHOLD).item()
        print('delete old thresholds...')
        os.remove(self.PYTHON_PATH_THRESHOLD + self.PYTHON_FILENAME_THRESHOLD)
        old_thresholds[name_model] = new_thres
        np.save(self.PYTHON_PATH_THRESHOLD + self.PYTHON_FILENAME_THRESHOLD, old_thresholds)

    def get_train_test(self,test_size=0.33):
        df = loadEmailsTrainData()
        df = df.loc[(df['body'].notnull()) & (df['Target'].notnull()), :]
        X = df['body'].astype(str).values
        y = df['Target'].astype(int).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test

    def fit_model(self,X,y,estimator,weights = [0.49,0.5]):
        sample_weights = (y == 0) * weights[0] + (y == 1) * weights[1]
        estimator.fit(X, y, **{'classifier__sample_weight': sample_weights} )
        return estimator

    def predict_target(self,X,name_model,estimator,threshold):
        y_score = estimator.predict_proba(X)
        y_pred = (y_score[:, 0] < threshold).astype(int)
        return y_pred

    def fit_grid_search(self,X,y,name_model='mnb',n_splits=3,n_iter=10):
        cv_dev = ShuffleSplit(n_splits=n_splits, test_size=0.33)

        scorer = make_scorer(accuracy_score)

        grid_search_estimator = self.weightEst(self.get_estimator(name_model),0.5, 0.5, 0.5)

        print(type(grid_search_estimator))
        print('checking estimator')
        print(grid_search_estimator.gridModel.named_steps['classifier'])
        print('did I see something?')
        print(grid_search_estimator.gridModel.named_steps['classifier'] is not None)
        grid_search = RandomizedSearchCV(estimator=grid_search_estimator,
                                         scoring=scorer,
                                         refit=True,
                                         cv=cv_dev,
                                         n_iter=n_iter,
                                         param_distributions={'w_0': norm(0.5, 0.1), 'w_1': norm(0.5, 0.1),
                                                              'thres': norm(0.5, 0.1)},
                                         verbose=4
                                         )

        print('Start Grid Search...')
        grid_search.fit(X, y)

        clf = grid_search.best_estimator_
        print('Best Parameters...')
        print(grid_search.best_params_)
        print('Best Score...')
        print(grid_search.best_score_)
        return {'opt_estimator':clf.gridModel,'opt_weight_taak':clf.w_0,'opt_weight_non_taak':clf.w_1,'opt_thres':clf.thres}
    ########################################################################################################################
    #                                      MODEL PROPERTIES                                                                #
    ########################################################################################################################

    def get_logProb(self,estimator,name_model,class_label):
        if (name_model == 'mnb'):
            logProb = estimator.named_steps['classifier'].feature_log_prob_
            if(class_label == 'NON_TAAK'):
                return logProb[1,:]
            elif(class_label == 'TAAK'):
                return logProb[0,:]
            else:
                return None
        elif(name_model == 'rf'):
            p = estimator.named_steps['classifier'].feature_importances_
            logProb = np.log( 1e-10 + p/np.sum(p) )
            return logProb
        elif(name_model == 'etr'):
            p = estimator.named_steps['classifier'].feature_importances_
            logProb = np.log( 1e-10 + p/np.sum(p) )
            return logProb
        else:
            return None

    def get_model_properties(self,estimator,name_model,class_label):
        log_probs = self.get_logProb(estimator,name_model,class_label)
        words_key = estimator.named_steps['vectorizer'].vocabulary_
        key_words = dict(zip([item[1] for item in words_key.items()],[item[0] for item in words_key.items()]))
        return log_probs,words_key,key_words

    def create_word_list(self,X_transformed, estimator, name_model, key_words):
        '''
        input: sparse matrix 1x n_words

        returns a list of tuples containing:
        _word
        _ exp( tfid[word]*log_probs[word|ham] )
        _  exp( tfid[word]*log_probs[word|spam] )
        '''
        if (name_model == 'mnb'):
            log_prob_taak =self.get_logProb(estimator, name_model, 'TAAK')
            log_prob_non_taak = self.get_logProb(estimator, name_model, 'NON_TAAK')
            print(log_prob_taak / log_prob_non_taak)
            indices_ = X_transformed.indices
            word_list = []
            for i in indices_:
                word_list += [
                    (key_words[i], X_transformed[0, i] * log_prob_taak[i], X_transformed[0, i] * log_prob_non_taak[i])]
            # if word_list is empty
            if len(word_list) == 0:
                word_list = [('', -1.0, -1.0)]
        elif (name_model == 'rf'):
            log_prob = self.get_logProb(estimator, name_model, '')
            indices_ = X_transformed.indices
            word_list = []
            for i in indices_:
                word_list += [(key_words[i], np.exp(log_prob[i]) * 100, np.exp(log_prob[i]) * 100)]
            # if word_list is empty
            if len(word_list) == 0:
                word_list = [('', -1.0, -1.0)]
        elif (name_model == 'etr'):
            log_prob = self.get_logProb(estimator, name_model, '')
            indices_ = X_transformed.indices
            word_list = []
            for i in indices_:
                word_list += [(key_words[i], np.exp(log_prob[i]) * 100, np.exp(log_prob[i]) * 100)]
            # if word_list is empty
            if len(word_list) == 0:
                word_list = [('', -1.0, -1.0)]
        return word_list

    def loadBestEstimator(self,name_model):
        filename_model = ''
        for filename in os.listdir(PYTHON_PATH_MODELS + name_model + '/'):
            if (filename.split('.')[1] == 'pkl'):
                filename_model = filename
                break
        if filename_model == '':
            return None
        else:
            with open(PYTHON_PATH_MODELS + name_model + '/' + filename_model, 'rb') as fid:
                estimator = pickle.load(fid)
            return estimator

    def getJsonEmailData(self,user):
        with open(self.PYTHON_PATH_JSON_INFO_EMAIL_IMAGES  + user + '/' + self.PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES ,
                  'r') as outfile:
            json_email_data = json.load(outfile)

        return json_email_data

    def deleteJsonEmailData(self, user):
        os.remove(self.PYTHON_PATH_JSON_INFO_EMAIL_IMAGES + user + '/' + self.PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES)

    def saveJsonEmailData(self, user,json_email_data):
        with open( self.PYTHON_PATH_JSON_INFO_EMAIL_IMAGES + user + '/' + self.PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES, 'w') as outfile:
            json.dump(json_email_data, outfile)



