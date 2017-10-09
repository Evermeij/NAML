#standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sqlite3
import pickle

import re

from flask import json
from wordcloud import WordCloud

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn.base import BaseEstimator

from scipy.stats import randint, expon, norm

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import make_scorer


from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer

from functools import partial,reduce
import itertools
import os


path_database = os.getcwd()+'/../webapp/static/data/databases/'
filename_database = 'database_NA_v1.db'

path_thresholds = os.getcwd()+ '/../webapp/static/Images/'
filename_thresholds = 'thresholds.npy'

path_confusion_matrix = os.getcwd() + '/../webapp/static/Images/confusion_matrices_NA/'
path_wordcloud = os.getcwd() + '/../webapp/static/Images/wordcloud_NA/'
path_pies = os.getcwd() + '/../webapp/static/Images/pies_NA/'
path_rocs = os.getcwd() + '/../webapp/static/Images/rocs_NA/'
path_models = os.getcwd()+'/../webapp/static/Images/models_NA/'
path_emails_feature_importance = os.getcwd() + '/../webapp/static/Images/Emails/feature_importance_email_NA/'
path_emails_pie_prob = os.getcwd() + '/../webapp/static/Images/Emails/pie_probability_NA/'

path_json_info_images = os.getcwd() + '/../webapp/static/Images/'
filename_json_info_images = 'json_filenames_NA.txt'

path_json_info_email_images = os.getcwd() + '/../webapp/static/Images/Emails/'
filename_json_info_email_images = 'json_email_data_NA.txt'


stopwords = set(stopwords.words('dutch'))
dutch_stemmer = stemmer = DutchStemmer()


def tokenize(text, stop=stopwords, stemmer=dutch_stemmer):
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


### CONFUSION MATRIX

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=8)
    plt.yticks(tick_marks, classes,fontsize=8)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_confusion_figs(model,name_model, X_test, y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(4, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=True)

    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=False)

    path = path_confusion_matrix
    file_name = name_model+'/'+'cm' + '.png'

    print('Saving  Confusion Matrix to '+path + file_name)
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()
    plt.close()
    return cnf_matrix

### CONFUSION MATRIX


def create_wordcloud(X,y,model,name_model):
    Xselect = X[y == 0]
    text = ''
    for body in list(Xselect):
        text += ' '.join(tokenize(body))

    wordcloud = WordCloud(width=300, height=200).generate(text)
    fig = plt.figure(figsize=(3.8, 3.8))
    plt.imshow(wordcloud)
    plt.axis("off")
    path = path_wordcloud
    file_name =  name_model+'/'+'wc'+ '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()
    plt.close()
### PIE

def create_pie(name_model,ntrain, ntest):
    labels = 'Training Set:\n ' + str(ntrain), 'Test Set:\n ' + str(ntest)
    sizes = [ntrain, ntest]
    colors = ['red', 'lightskyblue']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    fig = plt.figure(figsize=(3.6, 3.6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=45)

    path = path_pies
    file_name = name_model+'/'+'pie'+'.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()


### ROC

def create_ROC(name_model,cnf_matrix,y_test, y_pred,y_score):
    FP = cnf_matrix[1][0] / (cnf_matrix[1][0] + cnf_matrix[1][1])
    TP = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])
    cnf_matrix = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])

    fig = plt.figure(figsize=(2.3, 2.3))
    plt.plot(fpr, tpr, color='blue')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.plot([FP], [TP], 'ro')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = path_rocs
    file_name = name_model+'/'+'roc'+'.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

### model
def write_estimator(model,name_model):
    path = path_models
    file_name = name_model+'/'+'model'+'.pkl'
    with open(path+file_name, 'wb') as fid:
        pickle.dump(model,fid)

def generate_new_images_manual_fit(name_model ='mnb',thres = 0.5,weight_taak = 0.5,weight_non_taak =0.49):
    print('Clean Confusion Matrices...')
    clean_dir( path_confusion_matrix,extra_dir=name_model+'/' )
    print('Clean Wordcloud...')
    clean_dir(path_wordcloud,extra_dir=name_model+'/')
    print('Clean Pie...')
    clean_dir(path_pies,extra_dir=name_model+'/')
    print('Clean Roc...')
    clean_dir(path_rocs,extra_dir=name_model+'/')
    print('Clean Models...')
    clean_dir(path_models,extra_dir=name_model+'/')

    print('update Threshold dictionary')
    threshold_dic[name_model] = thres
    estimator = get_estimator(name_model)
    X_train, X_test, y_train, y_test = get_train_test(path_database, filename_database, test_size=0.3)
    estimator = fit_model(X_train, y_train, estimator,weights=[weight_taak,weight_non_taak])
    # X_test = X_test[:, 0] -> not necessary as long as only one column is being used for training

    ntrain = len(X_train)
    ntest = len(X_test)

    y_score = estimator.predict_proba(X_test)
    y_pred = (y_score[:, 0] < thres).astype(int)
    cnf_matrix = create_confusion_figs(estimator, name_model, X_test, y_test, y_pred)
    create_ROC(name_model, cnf_matrix, y_test, y_pred, y_score)
    create_pie(name_model,ntrain, ntest)
    create_wordcloud(X_train,y_train, estimator, name_model)
    write_estimator(estimator, name_model)

def get_new_predictions(ml_model,path_database, filename_database):
    update_dic = dict()
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,body FROM TABLE_MAILS ')
    for x in c:
        ypred = ml_model.predict(np.array([x[1]]))
        update_dic.update({str(x[0]):ypred})
    conn.close()
    return update_dic

def update_predictions(path_database, filename_database,update_dic):
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    for id,pred in update_dic.items():
        c.execute("UPDATE TABLE_MAILS\
                    SET pred_class = ?\
                    WHERE mail_id = ?",(str(pred[0]),id))
    conn.commit()
    conn.close()

def generate_new_images_auto_fit(name_model ='mnb'):
    print('Clean Confusion Matrices...')
    clean_dir( path_confusion_matrix,extra_dir=name_model+'/' )
    print('Clean Wordcloud...')
    clean_dir(path_wordcloud,extra_dir=name_model+'/')
    print('Clean Pie...')
    clean_dir(path_pies,extra_dir=name_model+'/')
    print('Clean Roc...')
    clean_dir(path_rocs,extra_dir=name_model+'/')
    print('Clean Models...')
    clean_dir(path_models,extra_dir=name_model+'/')

    estimator = get_estimator(name_model)
    X_train, X_test, y_train, y_test = get_train_test(path_database, filename_database, test_size=0.3)

    result_grid_search = fit_grid_search(X_train,y_train,name_model=name_model,n_splits=3,n_iter=10)
    estimator = result_grid_search['opt_estimator']

    set_threshold_dic(name_model, result_grid_search['opt_thres'])
    threshold_dic = get_threshold_dic()

    ntrain = len(X_train)
    ntest = len(X_test)

    y_score = estimator.predict_proba(X_test)
    y_pred = (y_score[:, 0] < threshold_dic[name_model]).astype(int)
    cnf_matrix = create_confusion_figs(estimator, name_model, X_test, y_test, y_pred)
    create_ROC(name_model, cnf_matrix, y_test, y_pred, y_score)
    create_pie(name_model,ntrain, ntest)
    create_wordcloud(X_train,y_train, estimator, name_model)
    write_estimator(estimator, name_model)

    print('Update predictions...')
    update_dic = get_new_predictions(estimator,path_database, filename_database)
    update_predictions(path_database, filename_database,update_dic)
    print('Predictions updated!')
########################################################################################################################


def create_word_list(X_transformed, estimator, name_model, key_words):
    '''
    input: sparse matrix 1x n_words

    returns a list of tuples containing:
    _word
    _ exp( tfid[word]*log_probs[word|ham] )
    _  exp( tfid[word]*log_probs[word|spam] )
    '''
    if (name_model == 'mnb'):
        log_prob_taak = get_logProb(estimator, name_model, 'TAAK')
        log_prob_non_taak = get_logProb(estimator, name_model, 'NON_TAAK')
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
        log_prob = get_logProb(estimator, name_model, '')
        indices_ = X_transformed.indices
        word_list = []
        for i in indices_:
            word_list += [(key_words[i], np.exp(log_prob[i]) * 100, np.exp(log_prob[i]) * 100)]
        # if word_list is empty
        if len(word_list) == 0:
            word_list = [('', -1.0, -1.0)]
    elif (name_model == 'etr'):
        log_prob = get_logProb(estimator, name_model, '')
        indices_ = X_transformed.indices
        word_list = []
        for i in indices_:
            word_list += [(key_words[i], np.exp(log_prob[i]) * 100, np.exp(log_prob[i]) * 100)]
        # if word_list is empty
        if len(word_list) == 0:
            word_list = [('', -1.0, -1.0)]
    return word_list


def return_html_body(body, word_list, y_pred, top_n_words=15):
    #### watch out for the dict -> needs to be moved ####
    def shorten_word(word,MAX_LEN=10):
        if len(word)>MAX_LEN:
            return word[:MAX_LEN]+'...'
        return word
    nhtml_classes = 3

    def get_html_class(n, nhtml_classes):
        if n > len(word_list) - top_n_words:
            return 2
        elif n > len(word_list) - 2 * top_n_words:
            return 1
        else:
            return 0

    min_len_sentence = 6
    max_len_sentence = 110
    max_sentences = 100
    spam_html_dic = {'color': {0: 'yellow', 1: 'orange', 2: 'red'}, 'size': {0: '18px', 1: '20px', 2: '28px'}}
    ham_html_dic = {'color': {0: 'lightblue', 1: 'blue', 2: 'darkblue'}, 'size': {0: '18px', 1: '20px', 2: '28px'}}

    # first replace \n and \r by corresponding html-flag
    # tokenize body
    tokens = (body.replace('\r', '').replace('\n', ' <br> ') ).encode('ascii', errors='ignore').decode('ascii').split()
    if (y_pred == 0):  # if ham
        # rank word_list by log_prob|ham
        sorted_word_list = sorted(word_list, key=lambda x: x[1])
        word_color_dic = dict(
            [(sorted_word_list[n][0], {'color': 'blue', 'size': ham_html_dic['size'][get_html_class(n, nhtml_classes)]})
             for n in range(len(word_list))])
    if (y_pred == 1):
        sorted_word_list = sorted(word_list, key=lambda x: x[2])
        word_color_dic = dict(
            [(sorted_word_list[n][0], {'color': 'red', 'size': spam_html_dic['size'][get_html_class(n, nhtml_classes)]})
             for n in range(len(word_list))])

        # construct html for words
    html_body = ''
    current_size_sentence = 0
    n_sentences = 0

    ##################################################################
   #small fix to avcoid getting empty text when there are too max <br>

    n_br = 0
    new_tokens = []
    for token in tokens:
        if '<br>' in token:
            n_br+=1
        if n_br >1:
            n_br = 0
        else:
            if len(new_tokens)==0:
                if '<br>' not in token:
                    new_tokens += [token]
            else:
                new_tokens += [token]

    tokens = new_tokens
    ###################################################################

    for word in tokens:
        if word.lower() in word_color_dic.keys():
            html_body += ' ' + '<span style="font-size:' + str(word_color_dic[word.lower()]['size']) + '">' + \
                         '<span style="color:' + str(word_color_dic[word.lower()]['color']) + '">' + str(
                word) + '</span>' + '</span>'
            current_size_sentence += len(word) + 1
        else:

            if (current_size_sentence > max_len_sentence):
                if ('<br>' in word):
                    #n_sub_sequent_br+=1
                    #if(n_sub_sequent_br)
                    html_body += word
                    current_size_sentence = 0

                    n_sentences += 1
                else:
                    html_body += '<br> ' + word
                    current_size_sentence = len(word)
                    n_sentences += 1
            else:
                if (current_size_sentence > min_len_sentence):
                    if ('<br>' in word):
                        html_body += ''
                    else:
                        html_body += ' ' + word
                        current_size_sentence += len(word) + 1
                else:
                    html_body += ' ' + word
                    if ('<br>' in word):
                        current_size_sentence = 0
                        n_sentences += 1
                    else:
                        current_size_sentence += len(word) + 1
        if n_sentences >= max_sentences:
            html_body += ' ...'
            return html_body
    return html_body


def create_feature_importance_email(name_model, word_list, extra_info, user, top_n_words=12):
    def shorten_word(word,MAX_LEN=10):
        if len(word)>MAX_LEN:
            return word[:MAX_LEN]+'...'
        return word
    fig = plt.figure(figsize=(7, 2))
    if (name_model == 'mnb'):
        top_ham = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        top_spam = sorted(word_list, key=lambda x: (x[2]))[:-top_n_words - 1:-1]
        # print(top_ham)
        max_score = -0.009 - np.max([item[1] for item in top_ham] + [item[1] for item in top_spam])
        min_score = -1.2 * np.min([item[1] for item in top_ham] + [item[1] for item in top_spam])
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(0, 3 * len(top_ham), 3), [min_score + score[1] for score in top_ham])
        plt.bar(np.arange(1, 3 * len(top_ham), 3), [min_score + score[2] for score in top_ham])
        plt.xticks(np.arange(0, 3 * len(top_ham), 3) + 0.5, [shorten_word(score[0]) for score in top_ham], rotation=45)
        plt.title('TAAK')
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(0, 3 * len(top_spam), 3), [min_score + score[1] for score in top_spam])
        plt.bar(np.arange(1, 3 * len(top_spam), 3), [min_score + score[2] for score in top_spam])
        plt.xticks(np.arange(0, 3 * len(top_spam), 3) + 0.5, [shorten_word(score[0]) for score in top_spam], rotation=45)
        plt.title('NON_TAAK')

        plt.legend(['prob. ham', 'prob. spam'])

        path = os.getcwd() + '/../webapp/static/Images/Emails/Users/'+user+'/feature_importance_email_NA/'+name_model + '/'

        file_name = 'efeature_imp_' + extra_info + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    elif (name_model == 'rf'):
        top_words = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        plt.bar(np.arange(0, len(top_words), 1), [score[1] for score in top_words])
        plt.xticks(np.arange(0, len(top_words), 1), [shorten_word(score[0]) for score in top_words], rotation=45)

        path = os.getcwd() + '/../webapp/static/Images/Emails/Users/'+user+'/feature_importance_email_NA/'+name_model + '/'
        file_name = 'efeature_imp_' + extra_info + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    elif (name_model == 'etr'):
        top_words = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        plt.bar(np.arange(0, len(top_words), 1), [score[1] for score in top_words])
        plt.xticks(np.arange(0, len(top_words), 1), [shorten_word(score[0]) for score in top_words], rotation=45)

        path = os.getcwd() + '/../webapp/static/Images/Emails/Users/'+user+'/feature_importance_email_NA/'+name_model + '/'
        file_name = 'efeature_imp_' + extra_info + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    else:
        pass


def create_prob_pie_email(name_model, score, extra_info, user,threshold=0.5):
    labels = 'TAAK', 'NON_TAAK'
    sizes = [score, 1.0 - score]
    colors = ['lightskyblue', 'red']
    explode = (0, 0.)
    # Plot
    fig = plt.figure(figsize=(2.5, 2.5))
    plt.pie(sizes, explode=explode, colors=colors, shadow=True)
    centre_circle = plt.Circle((0, 0), 0.75, color='black', fc='white', linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    if (score < threshold):
        center_text = 'NON_TAAK\n' + str(np.round((1. - score) * 100, 1)) + '%'
        plt.text(-0.4, -0.1, center_text, fontsize=12)
    else:
        center_text = 'TAAK\n' + str(np.round(score * 100, 1)) + '%'
        plt.text(-0.3, -0.1, center_text, fontsize=12)

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')

    path = os.getcwd() + '/../webapp/static/Images/Emails/Users/'+user+'/pie_probability_NA/'+name_model+'/'
    file_name = 'epie_prob_' +  extra_info + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

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
    #print('Confusion Matrices...')
    #clean_dir( path_confusion_matrix )
    #print('Wordcloud...')
    #clean_dir(path_wordcloud)
    #print('Pie...')
    #clean_dir(path_pies)
    #print('Roc...')
    #clean_dir(path_rocs)
    #print('Models...')
    #clean_dir(path_models)
    #print('Emails feature importance...')
    #clean_dir(path_emails_feature_importance)
    #print('Emails pie prob...')
    #clean_dir(path_emails_pie_prob)
    #print('jsons-data...')
    #clean_file(path_json_info_images,filename_json_info_images )
    #clean_file(path_json_info_email_images,filename_json_info_email_images )

    #print('Generate new images for the global performance...')
    #generate_new_images(thres=[0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,
    #     0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55])
    #print('Generate new images for images...')
    #generate_email_images(nmails=30)
