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


model_dict = {'mnb': MultinomialNB(fit_prior=False), 'rf': RandomForestClassifier(n_estimators=50),
              'etr': ExtraTreesClassifier(n_estimators=50)}

threshold_dic = {'mnb':0.44,'rf':0.49,'etr':0.392 }

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
    estimator.fit(X, y, **{'classifier__sample_weight': sample_weights})
    return estimator

def predict_target(X,name_model,estimator):
    th = threshold_dic[name_model]
    y_score = estimator.predict_proba(X)
    y_pred = (y_score[:, 0] < th).astype(int)
    return y_pred


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


def create_confusion_figs(model,name_model, X_test, y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(4, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=True)

    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=False)

    path = path_confusion_matrix
    file_name = name_model+'/'+'cm' + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()
    plt.close()
    return cnf_matrix

### CONFUSION MATRIX


def create_wordcloud(model,name_model):
    # Take 50 most import words, then normalize the log_prob
    Nwords = 50
    Ncopies = 300

    log_probs, words_key, key_words =  get_model_properties(model,name_model,class_label='TAAK')
    list_words_logprob = [(key_words[i], np.exp(log_probs[i])) for i in range(len(key_words.keys()))]
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
    plt.plot([FP], [TP], 'ro')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = path_rocs
    file_name = name_model+'/'+'roc'+'.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

### model
def write_estimator(model,name_model,t):
    path = path_models
    file_name = name_model+'/'+'model'+'.pkl'
    with open(path+file_name, 'wb') as fid:
        pickle.dump(model,fid)

def generate_new_images_from(name_model ='mnb',thres = 0.5,weight_taak = 0.5,weight_non_taak =0.49):
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
    create_wordcloud(estimator, name_model)
    write_estimator(estimator, name_model)

def generate_new_images(nsizes = [1500],ratios = [0.3],thres = [0.4920, 0.4925, 0.4930, 0.4935 ,0.4940]):
    ind_train = 0
    filenames = []
    for name_model in model_dict.keys():
        temp_s = []
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
                    estimator = get_estimator(name_model)
                    X_train, X_test, y_train, y_test = get_train_test(path_database,filename_database,test_size=0.3)
                    estimator = fit_model(X_train,y_train,estimator)
                    #X_test = X_test[:, 0] -> not necessary as long as only one column is being used for training

                    y_score = estimator.predict_proba(X_test)
                    y_pred = (y_score[:, 0] < th).astype(int)
                    cnf_matrix = create_confusion_figs(estimator, name_model, X_test, y_test, y_pred, ntrain, ntest,
                                                       ind_train, ind_test)
                    create_ROC(name_model, cnf_matrix, ntrain, ntest, ind_train, ind_test, y_test, y_pred, y_score)
                    create_pie(name_model, ntrain, ntest, ind_train, ind_test)
                    create_wordcloud(estimator, name_model, ntrain, ntest, ind_train, ind_test)

                    write_estimator(estimator,name_model, ntrain, ntest, ind_train, ind_test)
                    extension = str(ntrain + ntest) + '_' + str(ntrain) + '_' + str(ntest) + '_' + str(
                        ind_train) + '_' + str(ind_test)
                    temp_th += [ \
                        {"cm": "static/Images/confusion_matrices_NA/" + name_model + '_' + 'cm_' + extension + '.png', \
                         "pie": "static/Images/pies_NA/" + name_model + '_' + "pie_" + extension + '.png', \
                         "roc": "static/Images/rocs_NA/" + name_model + '_' + "roc_" + extension + '.png', \
                         "wc": "static/Images/wordcloud_NA/" + name_model + '_' + "wc_" + extension + '.png', \
                         "md": "static/models/" + name_model + '_' + "model_NA_" + extension + '.pkl'} \
                        ]
                temp_r += [temp_th]
            temp_s += [temp_r]
        filenames += [temp_s]
    with open(os.getcwd() + '/../webapp/static/Images/' + 'json_filenames_NA.txt', 'w') as outfile:
        json.dump(filenames, outfile)
    print('Done!')






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
        if word.lower() in word_color_dic.keys():
            html_body += ' ' + '<span style="font-size:' + str(word_color_dic[word.lower()]['size']) + '">' + \
                         '<span style="color:' + str(word_color_dic[word.lower()]['color']) + '">' + str(
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


def create_feature_importance_email(name_model, word_list, ind_email, top_n_words=12):
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
        plt.xticks(np.arange(0, 3 * len(top_ham), 3) + 0.5, [score[0] for score in top_ham], rotation=45)
        plt.title('TAAK')
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(0, 3 * len(top_spam), 3), [min_score + score[1] for score in top_spam])
        plt.bar(np.arange(1, 3 * len(top_spam), 3), [min_score + score[2] for score in top_spam])
        plt.xticks(np.arange(0, 3 * len(top_spam), 3) + 0.5, [score[0] for score in top_spam], rotation=45)
        plt.title('NON_TAAK')

        plt.legend(['prob. ham', 'prob. spam'])

        path = os.getcwd() + '/../webapp/static/Images/Emails/feature_importance_email_NA/'
        file_name = name_model + '_' + 'efeature_imp_' + str(ind_email) + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    elif (name_model == 'rf'):
        top_words = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        plt.bar(np.arange(0, len(top_words), 1), [score[1] for score in top_words])
        plt.xticks(np.arange(0, len(top_words), 1), [score[0] for score in top_words], rotation=45)

        path = os.getcwd() + '/../webapp/static/Images//Emails/feature_importance_email_NA/'
        file_name = name_model + '_' + 'efeature_imp_' + str(ind_email) + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    elif (name_model == 'etr'):
        top_words = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        plt.bar(np.arange(0, len(top_words), 1), [score[1] for score in top_words])
        plt.xticks(np.arange(0, len(top_words), 1), [score[0] for score in top_words], rotation=45)

        path = os.getcwd() + '/../webapp/static/Images//Emails/feature_importance_email_NA/'
        file_name = name_model + '_' + 'efeature_imp_' + str(ind_email) + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    else:
        pass


def create_prob_pie_email(name_model, score, ind_email, threshold=0.5):
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
    else:
        center_text = 'TAAK: ' + str(np.round(score * 100, 1)) + '%'

    plt.text(-0.4, 0.0, center_text, fontsize=10)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')

    path = os.getcwd() + '/../webapp/static/Images//Emails/pie_probability_NA/'
    file_name = name_model + '_' + 'epie_prob_' + str(ind_email) + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

def generate_email_images(nmails = 15):
    spam_ham_dic = {0: 'TAAK', 1: 'NON_TAAK'}


    df = get_n_mails_of(path_database, filename_database, nmails= nmails)
    df = df.loc[df['body'].notnull(), :]
    X_test = df[['body', 'Date', 'From', 'Subject']].values
    y_test = df['Target'].astype(int).values
    json_email_data = []

    for name_model in model_dict.keys():
        for filename in os.listdir(path_models+name_model+'/'):
            if ( filename.split('.')[1]== 'pkl'):
                filename_model = filename
                break


        with open(path_models + name_model+'/'+filename_model, 'rb') as fid:
            estimator = pickle.load(fid)

        log_probs, words_key, key_words = get_model_properties(estimator, name_model, 'TAAK')

        list_email_ind = range(len(y_test))

        email_data = []
        for ind_test in list_email_ind:
            n_test = ind_test
            body = X_test[n_test:n_test + 1, 0]
            date = X_test[n_test:n_test + 1, 1]
            _from = X_test[n_test:n_test + 1, 2]
            subject = X_test[n_test:n_test + 1, 3]
            X_transformed = estimator.named_steps['vectorizer'].transform(body)
            indices_ = X_transformed.indices

            score = estimator.predict_proba(body)
            y_pred = int(score[0][0] < threshold_dic[name_model])
            print(score)
            print(y_pred, y_test[ind_test])

            word_list = create_word_list(X_transformed, estimator, name_model, key_words)
            print(X_transformed.shape)

            html_body = return_html_body(body[0], word_list, y_pred, top_n_words=8)
            create_prob_pie_email(name_model, score[0][0], 'email_' + str(ind_test))
            create_feature_importance_email(name_model, word_list, 'email_' + str(ind_test), top_n_words=12)
            email_data += [{'pred': spam_ham_dic[y_pred],
                            'truth': spam_ham_dic[y_test[ind_test]],
                            'date': date[0],
                            'from': _from[0],
                            'subject': subject[0],
                            'html_body': html_body,
                            'eFimp': "/static/Images/Emails/feature_importance_email_NA/" + name_model + '_' + "efeature_imp_" + 'email_' + str(
                                ind_test) + '.png',
                            'epie': "/static/Images/Emails/pie_probability_NA/" + name_model + '_' + "epie_prob_" + 'email_' + str(
                                ind_test) + '.png'}]
        json_email_data += [email_data]
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
    print('Generate new images for images...')
    generate_email_images(nmails=30)
