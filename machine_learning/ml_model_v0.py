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

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer

from functools import partial,reduce
import itertools
import os


path_database = os.getcwd()+'/../webapp/static/data/databases/'
filename_database = 'database_NA_v1.db'

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


estimator = Pipeline(steps=[

    ('vectorizer',
     TfidfVectorizer(input=u'content', encoding=u'latin1', decode_error=u'strict', strip_accents=None, lowercase=True,
                     preprocessor=None, tokenizer=partial(tokenize, stop=stopwords, stemmer=dutch_stemmer),
                     analyzer=u'word',  # stop_words=(stopwords.words('dutch')),
                     ngram_range=(1, 3),  # max_df=0.9, min_df=0.005,
                     max_features=10000, vocabulary=None, binary=False,
                     norm=u'l1', use_idf=True, smooth_idf=True, sublinear_tf=False)),

    ('classifier', MultinomialNB(fit_prior=False))
]
)
th = 0.4925

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
def get_n_mails_of(path_database,filename_database, nmails=10, address=''):
    index_ = 0
    df = pd.DataFrame()
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,body,truth_class FROM TABLE_MAILS ')
    for x in c:
        df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2]}, index=[index_])])
        index_ += 1
        if(index_>=nmails):
            break
    conn.close()
    return df

def fit_model(X,y,estimator):
    sample_weights = (y == 0) * 0.49 + (y == 1) * 0.5
    estimator.fit(X, y, **{'classifier__sample_weight': sample_weights})

def predict_target(X,estimator):
    y_score = estimator.predict_proba(X)
    y_pred = (y_score[:, 0] < th).astype(int)
    return y_pred


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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_confusion_figs(model, X_test, y_test, y_pred, ntrain, ntest, ind_train, ind_test):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(4, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=True)

    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=False)

    path = path_confusion_matrix
    file_name = 'cm_' + str(ntrain + ntest) + '_' + str(ntrain) + '_' + str(ntest) + '_' + str(ind_train) + '_' + str(
        ind_test) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()
    plt.close()
    return cnf_matrix

### CONFUSION MATRIX

def get_model_properties(estimator):
    log_probs = estimator.named_steps['classifier'].feature_log_prob_
    words_key = estimator.named_steps['vectorizer'].vocabulary_
    key_words = dict(zip([item[1] for item in words_key.items()],[item[0] for item in words_key.items()]))
    return log_probs,words_key,key_words


def create_wordcloud(model, ntrain, ntest, ind_train, ind_test):
    # Take 50 most import words, then normalize the log_prob
    Nwords = 50
    Ncopies = 300

    log_probs, words_key, key_words = get_model_properties(model)
    list_words_logprob = [(key_words[i], np.exp(log_probs[1, i])) for i in range(len(key_words.keys()))]
    log_top = sorted(list_words_logprob, key=lambda x: x[1])[:-Nwords:-1]
    sum_log_prob = reduce(lambda x, y: x + y[1], log_top, 0)
    log_top = [(x[0], x[1] / sum_log_prob) for x in log_top]
    text = []
    for item in log_top:
        text += [item[0]] * int(item[1] * Ncopies)

    wordcloud = WordCloud(width=300, height=200).generate(' '.join(text))
    fig = plt.figure(figsize=(3.8, 3.8))
    plt.imshow(wordcloud)
    plt.axis("off")
    path = path_wordcloud
    file_name = 'wc_' + str(ntrain + ntest) + '_' + str(ntrain) + '_' + str(ntest) + '_' + str(ind_train) + '_' + str(
        ind_test) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()

### PIE

def create_pie(ntrain, ntest, ind_train, ind_test):
    labels = 'Training Set:\n ' + str(ntrain), 'Test Set:\n ' + str(ntest)
    sizes = [ntrain, ntest]
    colors = ['red', 'lightskyblue']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    fig = plt.figure(figsize=(3.6, 3.6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=45)

    path = path_pies
    file_name = 'pie_' + str(ntrain + ntest) + '_' + str(ntrain) + '_' + str(ntest) + '_' + str(ind_train) + '_' + str(
        ind_test) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()


### ROC

def create_ROC(cnf_matrix, ntrain, ntest, ind_train, ind_test,y_test, y_pred,y_score):
    FP = cnf_matrix[0, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1])
    TP = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    cnf_matrix = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])

    fig = plt.figure(figsize=(2.3, 2.3))
    plt.plot(fpr, tpr, color='blue')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([FP], [TP], 'ro')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = path_rocs
    file_name = 'roc_' + str(ntrain + ntest) + '_' + str(ntrain) + '_' + str(ntest) + '_' + str(ind_train) + '_' + str(
        ind_test) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

### model
def write_estimator(model,ntrain,ntest,ind_train,ind_test):
    path = path_models
    file_name = 'model_'+str(ntrain+ntest)+'_'+str(ntrain)+'_'+str(ntest)+'_'+str(ind_train)+'_'+str(ind_test)+'.pkl'
    with open(path+file_name, 'wb') as fid:
        pickle.dump(model,fid)

def generate_new_images(nsizes = [1980],ratios = [0.3],thres = [0.4920, 0.4925, 0.4930, 0.4935 ,0.4940]):
    ind_train = 0
    filenames = []
    for nsize in nsizes:
        ind_train += 1
        ind_test = 0
        temp_r = []
        for ratio in ratios:
            temp_th = []
            for th in thres:
                ind_test += 1
                ntest = int(ratio * nsize)
                ntrain = int(nsize - ntest)

                X_train, X_test, y_train, y_test = get_train_test(path_database,filename_database,test_size=0.3)

                nsize = len(X_train) +  len(X_test)

                ntest = int(ratio * nsize)
                ntrain = int(nsize - ntest)
                fit_model(X_train, y_train)
                # y_pred = estimator.predict(X_test)
                y_score = estimator.predict_proba(X_test)
                y_pred = (y_score[:, 0] < th).astype(int)
                cnf_matrix = create_confusion_figs(estimator, X_test, y_test, y_pred, ntrain, ntest, ind_train,
                                                   ind_test)
                create_ROC(cnf_matrix, ntrain, ntest, ind_train, ind_test, y_test, y_pred, y_score)
                create_pie(ntrain, ntest, ind_train, ind_test)
                create_wordcloud(estimator, ntrain, ntest, ind_train, ind_test)

                write_estimator(estimator, ntrain, ntest, ind_train, ind_test)
                extension = str(ntrain + ntest) + '_' + str(ntrain) + '_' + str(ntest) + '_' + str(
                    ind_train) + '_' + str(ind_test)
                temp_th += [ \
                    {"cm": "static/Images/confusion_matrices_NA/cm_" + extension + '.png', \
                     "pie": "static/Images/pies_NA/pie_" + extension + '.png', \
                     "roc": "static/Images/rocs_NA/roc_" + extension + '.png', \
                     "wc": "static/Images/wordcloud_NA/wc_" + extension + '.png', \
                     "md": "static/models/model_NA_" + extension + '.pkl'} \
                    ]
            temp_r += [temp_th]
        filenames += [temp_r]
    with open(os.getcwd() + '/../webapp/static/Images/' + 'json_filenames_NA.txt', 'w') as outfile:
        json.dump(filenames, outfile)






########################################################################################################################


def create_word_list(X_transformed, log_probs, key_words):
    '''
    input: sparse matrix 1x n_words

    returns a list of tuples containing:
    _word
    _ exp( tfid[word]*log_probs[word|ham] )
    _  exp( tfid[word]*log_probs[word|spam] )
    '''
    indices_ = X_transformed.indices
    word_list = []
    for i in indices_:
        word_list += [(key_words[i], np.array(np.array(X_transformed[0, i] * log_probs[0, i])),
                       np.array(np.array(X_transformed[0, i] * log_probs[1, i])))]
    # if word_list is empty
    if len(word_list) == 0:
        word_list = [('', -1.0, -1.0)]
    return word_list


def return_html_body(body, word_list, y_pred, top_n_words=15):
    #### watch out for the dict -> needs to be moved ####

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
    max_sentences = 8
    spam_html_dic = {'color': {0: 'yellow', 1: 'orange', 2: 'red'}, 'size': {0: '18px', 1: '20px', 2: '28px'}}
    ham_html_dic = {'color': {0: 'lightblue', 1: 'blue', 2: 'darkblue'}, 'size': {0: '18px', 1: '20px', 2: '28px'}}

    # first replace \n and \r by corresponding html-flag
    # tokenize body
    tokens = body.replace('\r', '').replace('\n', ' <br> ').split()
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
    for word in tokens:
        if word in word_color_dic.keys():
            html_body += ' ' + '<span style="font-size:' + str(word_color_dic[word]['size']) + '">' + \
                         '<span style="color:' + str(word_color_dic[word]['color']) + '">' + str(
                word) + '</span>' + '</span>'
            current_size_sentence += len(word) + 1
        else:

            if (current_size_sentence > max_len_sentence):
                if ('<br>' in word):
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


def create_feature_importance_email(word_list, ind_email, top_n_words=12):
    top_ham = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
    top_spam = sorted(word_list, key=lambda x: (x[2]))[:-top_n_words - 1:-1]
    # print(top_ham)
    max_score = -0.009 - np.max([item[1] for item in top_ham] + [item[1] for item in top_spam])
    min_score = -1.2 * np.min([item[1] for item in top_ham] + [item[1] for item in top_spam])
    fig = plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(0, 3 * len(top_ham), 3), [min_score + score[1] for score in top_ham])
    plt.bar(np.arange(1, 3 * len(top_ham), 3), [min_score + score[2] for score in top_ham])
    plt.xticks(np.arange(0, 3 * len(top_ham), 3) + 0.5, [score[0] for score in top_ham], rotation=45)
    plt.title('TAAK')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(0, 3 * len(top_spam), 3), [min_score + score[1] for score in top_spam])
    plt.bar(np.arange(1, 3 * len(top_spam), 3), [min_score + score[2] for score in top_spam])
    plt.xticks(np.arange(0, 3 * len(top_spam), 3) + 0.5, [score[0] for score in top_spam], rotation=45)
    plt.title('NON_TAAK')

    plt.legend(['prob. ham', 'prob. spam'])

    path = path_emails_feature_importance
    file_name = 'efeature_imp_' + str(ind_email) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()


def create_prob_pie_email(score, ind_email, threshold=0.5):
    labels = 'TAAK', 'NON_TAAK'
    sizes = [score, 1.0 - score]
    colors = ['lightskyblue', 'red']
    explode = (0, 0.)
    # Plot
    fig = plt.figure(figsize=(2.5, 2.5))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True)
    centre_circle = plt.Circle((0, 0), 0.75, color='black', fc='white', linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    if (score < threshold):
        center_text = 'NON_TAAK: ' + str(np.round((1. - score) * 100, 1)) + '%'
        plt.text(-0.6, 0.0, center_text, fontsize=8)
    else:
        center_text = 'TAAK: ' + str(np.round(score * 100, 1)) + '%'
        plt.text(-0.45, 0.0, center_text, fontsize=8)

    plt.axis('equal')

    path = path_emails_pie_prob
    file_name = 'epie_prob_' + str(ind_email) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

def generate_email_images(nmails = 10):
    spam_ham_dic = {0: 'TAAK', 1: 'NON_TAAK'}


    df = get_n_mails_of(path_database, filename_database, nmails= nmails)
    for filename in os.listdir(path_models):
        if (filename.split('.')[1]== 'pkl'):
            filename_model = filename
            break


    with open(path_models + filename_model, 'rb') as fid:
        estimator = pickle.load(fid)
    X_test = df['body'].astype(str).values
    y_test = df['Target'].astype(int).values

    log_probs, words_key, key_words = get_model_properties(estimator)
    y_preds = estimator.predict(X_test)
    FP_indices = np.array(range(X_test.shape[0]))[y_test != y_preds]
    list_email_ind = list(set([14, 15,16, 17, 18, 19, 20,21,22,23]) )# + list(FP_indices[0:1])))
    json_email_data = []
    for ind_test in list_email_ind:
        n_test = ind_test
        body = X_test[n_test:n_test + 1][0]
        X_transformed = estimator.named_steps['vectorizer'].transform(X_test[n_test:n_test + 1])
        indices_ = X_transformed.indices
        y_pred = estimator.predict(X_test[n_test:n_test + 1])
        score = estimator.predict_proba(X_test[n_test:n_test + 1])
        print(score)
        print(y_pred, y_test[ind_test])

        word_list = create_word_list(X_transformed, log_probs, key_words)
        print(X_transformed.shape)
        html_body = return_html_body(body, word_list, y_pred[0], top_n_words=12)
        create_prob_pie_email(score[0][0], 'email' + str(ind_test))
        create_feature_importance_email(word_list, 'email' + str(ind_test), top_n_words=12)

        json_email_data += [{'pred': spam_ham_dic[y_pred[0]],
                             'truth': spam_ham_dic[y_test[ind_test]],
                             'html_body': html_body,
                             'eFimp': "/static/Images/Emails/feature_importance_email_NA/efeature_imp_" + 'email' + str(
                                 ind_test) + '.png',
                             'epie': "/static/Images/Emails/pie_probability_NA/epie_prob_" + 'email' + str(
                                 ind_test) + '.png'}]

    with open( path_json_info_email_images + filename_json_info_email_images, 'w') as outfile:
        json.dump(json_email_data, outfile)


### OTHER FUNCTIONS
def clean_dir(pathdir):
    '''
    :param pathdir: 
    :return: deletes all .png and .txt within the dir
    '''
    for filename in os.listdir(pathdir):
        if (filename.split('.')[1]== 'txt') or (filename.split('.')[1]== 'png')or (filename.split('.')[1]== 'pkl'):
            print('Deleting File: '+str(filename) )
            os.remove(pathdir+filename)
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
    print('Confusion Matrices...')
    clean_dir( path_confusion_matrix )
    print('Wordcloud...')
    clean_dir(path_wordcloud)
    print('Pie...')
    clean_dir(path_pies)
    print('Roc...')
    clean_dir(path_rocs)
    print('Models...')
    clean_dir(path_models)
    print('Emails feature importance...')
    clean_dir(path_emails_feature_importance)
    print('Emails pie prob...')
    clean_dir(path_emails_pie_prob)
    print('jsons-data...')
    clean_file(path_json_info_images,filename_json_info_images )
    clean_file(path_json_info_email_images,filename_json_info_email_images )

    print('Generate new images...')
    generate_new_images()
    print('Generate new images...')
    generate_email_images(nmails=30)
