#standard packages
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.use('Agg')

import sqlite3
import pickle

import re

from flask import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn.base import BaseEstimator

from scipy.stats import randint, expon, norm


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import make_scorer


from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer

from functools import partial,reduce
import os

localdir = '/app'
path_database = localdir + '/static/data/databases/'
filename_database = 'database_NA_v1.db'

path_thresholds =localdir + '/static/Images/'
filename_thresholds = 'thresholds.npy'

path_confusion_matrix = localdir + '/static/Images/confusion_matrices_NA/'
path_wordcloud = localdir + '/static/Images/wordcloud_NA/'
path_pies = localdir + '/static/Images/pies_NA/'
path_rocs = localdir + '/static/Images/rocs_NA/'
path_models = localdir + '/static/Images/models_NA/'
path_emails_feature_importance = localdir + '/static/Images/Emails/feature_importance_email_NA/'
path_emails_pie_prob = localdir + '/static/Images/Emails/pie_probability_NA/'

path_info_images = localdir + '/static/Images/'
filename_info_images = 'filenames_imagesNA.npy'

path_json_info_email_images = localdir + '/static/Images/Emails/'
filename_json_info_email_images = 'json_email_data_NA.txt'

path_user_email_images = localdir + '/static/Images/Emails/Users/'



def tokenize(text, stop, stemmer):
    """Converts text to tokens."""
    # tokens = word_tokenize(text, language='dutch')

    tokens = [word.lower() for word in text.split()]
    tokens = [i for i in tokens if i not in stop]
    tokens = ["".join(re.findall("[a-zA-Z]+", word)) for word in tokens]
    tokens = list(filter(lambda x: len(x) > 2, tokens))
    # tokens = [stemmer.stem(word) for word in tokens]
    return tokens


model_dict = {'mnb': MultinomialNB(fit_prior=False), 'rf': RandomForestClassifier(n_estimators=50),
              'etr': ExtraTreesClassifier(n_estimators=50)}

def load_filenames_images():
    filenames_dict = np.load(path_info_images+filename_info_images).item()
    return filenames_dict
def get_threshold_dic():
    return np.load(path_thresholds+filename_thresholds ).item()
def set_threshold_dic(name_model,new_thres):
    old_thresholds = np.load(path_thresholds+filename_thresholds ).item()
    print('delete old thresholds...')
    os.remove(path_thresholds+filename_thresholds)
    old_thresholds[name_model] = new_thres
    np.save(path_thresholds+filename_thresholds , old_thresholds)


threshold_dic = get_threshold_dic()
def get_estimator(model_name = 'mnb'):
    stopwords = set(stopwords.words('dutch'))
    dutch_stemmer = stemmer = DutchStemmer()

    model = model_dict[model_name]
    estimator = Pipeline(steps=[

        ('vectorizer', TfidfVectorizer(input=u'content', encoding=u'latin1', decode_error=u'strict', strip_accents=None,
                                       lowercase=True,
                                       preprocessor=None,
                                       tokenizer=partial(tokenize, stop=stopwords, stemmer=dutch_stemmer),
                                       analyzer=u'word',  # stop_words=(stopwords.words('dutch')),
                                       ngram_range=(1, 3),  # max_df=0.9, min_df=0.005,
                                       max_features=10000, vocabulary=None, binary=False,
                                       norm=u'l1', use_idf=True, smooth_idf=True, sublinear_tf=False)),

        ('classifier', model)
        ]
        )
    return estimator

def get_train_test(path_database,filename_database,test_size=0.3):
    df = pd.DataFrame()
    index_ = 0
    conn = sqlite3.connect(path_database+filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,body,truth_class FROM TABLE_MAILS ')
    for x in c:
        df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2]}, index=[index_])])
        index_ += 1
    conn.close()
    df = df.loc[(df['body'].notnull()) & (df['Target'].notnull()), :]
    X = df['body'].astype(str).values
    y = df['Target'].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test
#
# def get_mail_test(mail_id):
#     X,y,df = get_mail(path_database,filename_database,mail_id)
#     return df

def get_mail(path_database,filename_database,mail_id):
    df = pd.DataFrame()
    index_ = 0
    conn = sqlite3.connect(path_database+filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,body,truth_class,date_sent,from_email_address,subject FROM TABLE_MAILS where mail_id=?',[(mail_id)])
    for x in c:
        df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2], 'Date':x[3],'From':x[4],'Subject':x[5]}, index=[index_])])
        index_ += 1
        break
    conn.close()
    X = df['body'].astype(str).values

    target = df['Target'].astype(str).values[0]


    return X,target,df


def get_n_mails_of(path_database,filename_database, nmails=10, address=''):
    index_ = 0
    df = pd.DataFrame()
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,body,truth_class,date_sent,from_email_address,subject FROM TABLE_MAILS ')
    for x in c:
        df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2], 'Date':x[3],'From':x[4],'Subject':x[5]}, index=[index_])])
        index_ += 1
        if(index_>=nmails):
            break
    conn.close()

    print(df.columns)
    return df


def fit_model(X,y,estimator,weights = [0.49,0.5]):
    sample_weights = (y == 0) * weights[0] + (y == 1) * weights[1]
    estimator.fit(X, y, **{'classifier__sample_weight': sample_weights} )
    return estimator

def predict_target(X,name_model,estimator):
    th = threshold_dic[name_model]
    y_score = estimator.predict_proba(X)
    y_pred = (y_score[:, 0] < th).astype(int)
    return y_pred

def fit_grid_search(X,y,name_model='mnb',n_splits=3,n_iter=10):
    class weightEst(BaseEstimator):
        def __init__(self, w_0, w_1, thres):
            self.w_0 = w_0
            self.w_1 = w_1
            self.thres = thres
            self.estimator = get_estimator(name_model)

        def fit(self, X, y):
            weight = self.w_0 * (y == 0) + self.w_1 * (y == 1)
            self.estimator.fit(X, y, **{'classifier__sample_weight': weight} )
            return self

        def predict(self, X):
            score = self.estimator.predict_proba(X)
            ypred = (score[:, 0] < self.thres).astype(int)
            return ypred

        def predict_proba(self, X):
            score = self.estimator.predict_proba(X)
            return score

        def get_params(self, deep=True):
            params = {'w_0': self.w_0, 'w_1': self.w_1, 'thres': self.thres}
            return params

        def set_params(self, **params):
            self.w_0 = params['w_0']
            self.w_1 = params['w_1']
            self.thres = params['thres']
            return self

    estimator = weightEst(0.5, 0.5, 0.5)

    cv_dev = ShuffleSplit(n_splits=n_splits, test_size=0.3)

    scorer = make_scorer(accuracy_score)
    grid_search = RandomizedSearchCV(estimator,
                                     scoring=scorer,
                                     refit=True,
                                     cv=cv_dev,
                                     n_iter=n_iter,
                                     param_distributions={'w_0': norm(0.5, 0.1), 'w_1': norm(0.5, 0.1),
                                                          'thres': norm(0.5, 0.1)},
                                     verbose=4
                                     )

    grid_search.fit(X, y)

    clf = grid_search.best_estimator_
    print('Best Parameters...')
    print(grid_search.best_params_)
    print('Best Score...')
    print(grid_search.best_score_)
    return {'opt_estimator':clf.estimator,'opt_weight_taak':clf.w_0,'opt_weight_non_taak':clf.w_1,'opt_thres':clf.thres}
########################################################################################################################
#                                      MODEL PROPERTIES                                                                #
########################################################################################################################

def get_logProb(estimator,name_model,class_label):
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

def get_model_properties(estimator,name_model,class_label):
    log_probs = get_logProb(estimator,name_model,class_label)
    words_key = estimator.named_steps['vectorizer'].vocabulary_
    key_words = dict(zip([item[1] for item in words_key.items()],[item[0] for item in words_key.items()]))
    return log_probs,words_key,key_words


########################################################################################################################
#                                               FIGURES                                                                #
########################################################################################################################




def add_new_email_images(mail_id,user='Mette'):
    spam_ham_dic = {'0': 'TAAK', '1': 'NON_TAAK'}
    def shorten_word(word,MAX_LEN=35):
        if len(word)>MAX_LEN:
            return word[:MAX_LEN]+'...'
        return word

    with open(path_json_info_email_images + "Users/"+user+'/'+ filename_json_info_email_images, 'r') as outfile:
        json_email_data = json.load(outfile)

    print(json_email_data.keys())
    X,target,df =  get_mail(path_database, filename_database, mail_id)

    for name_model in model_dict.keys():
        for filename in os.listdir(path_models+name_model+'/'):
            if ( filename.split('.')[1]== 'pkl'):
                filename_model = filename
                break


        with open(path_models + name_model+'/'+filename_model, 'rb') as fid:
            estimator = pickle.load(fid)

        log_probs, words_key, key_words = get_model_properties(estimator, name_model, 'TAAK')



        body = X
        date = df['Date']
        _from = df['From']
        subject = df['Subject']
        X_transformed = estimator.named_steps['vectorizer'].transform(body)

        word_list = create_word_list(X_transformed, estimator, name_model, key_words)

        score = estimator.predict_proba(body)
        y_pred = int(score[0][0] < threshold_dic[name_model])


        print(X_transformed.shape)

        html_body = return_html_body(body[0], word_list, y_pred, top_n_words=20)
        extra_info = 'email_' + mail_id.replace('.','').replace('>','').replace('<','').replace('/','').replace('\\','')
        create_prob_pie_email(name_model, score[0][0], extra_info , user,threshold_dic[name_model])
        create_feature_importance_email(name_model, word_list, extra_info ,user, top_n_words=5)

        print('here...')
        #print(y)
        email_data = {'pred': spam_ham_dic[str(y_pred)],
                        'truth': spam_ham_dic.get(target,'NONE'),
                        'date': date[0],
                        'from': _from[0],
                        'subject': shorten_word(subject[0]),
                        'html_body': html_body,
                        'eFimp': "/static/Images/Emails/Users/"+user+'/feature_importance_email_NA/' + name_model + '/' + "efeature_imp_" + extra_info + '.png',
                        'epie': "/static/Images/Emails/Users/" +user+'/pie_probability_NA/'+ name_model + '/' + "epie_prob_" + extra_info + '.png'}
        if name_model not in json_email_data.keys():
            json_email_data[name_model] = list([email_data])
        else:
            json_email_data[name_model]+= [email_data]
    print('Remove old file...')
    os.remove(path_json_info_email_images + "Users/"+user+'/'+ filename_json_info_email_images)
    print('Create new file')
    with open( path_json_info_email_images + "Users/"+user+'/'+ filename_json_info_email_images, 'w') as outfile:
        json.dump(json_email_data, outfile)


def clean_dir(pathdir,extra_dir = ''):
    '''
    :param pathdir: 
    :return: deletes all .png and .txt within the dir
    '''
    for filename in os.listdir(pathdir+extra_dir):
        if (filename.split('.')[1]== 'txt') or (filename.split('.')[1]== 'png')or (filename.split('.')[1]== 'pkl'):
            print('Deleting File: '+str(filename) )
            os.remove(pathdir+extra_dir+filename)
def clean_file(pathdir,selectFilename):
    '''
    :param pathdir: 
    :return: deletes all .png and .txt within the dir
    '''
    for filename in os.listdir(pathdir):
        if (filename== selectFilename):
            print('Deleting File: '+str(filename) )
            os.remove(pathdir+filename)
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_train_test(path_database, filename_database, test_size=0.3)

    fit_grid_search(X_train,y_train,name_model='etr',n_splits=3,n_iter=10)

