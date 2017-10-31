
import numpy as np

import os,sys
import sqlite3
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score

from wordcloud import WordCloud

import os,sys
import itertools
from functools import partial,reduce

#------ PATHS

#--- LOCAL PATHS FOR LOADING IMAGES FOR THE JS CONTROLLER
JS_PATH_CONFUSION_MATRIX = '/static/Images/confusion_matrices_NA/'
JS_PATH_WORDCLOUD = '/static/Images/wordcloud_NA/'
JS_PATH_PIES = '/static/Images/pies_NA/'
JS_PATH_ROCS = '/static/Images/rocs_NA/'
JS_PATH_MODELS = '/static/Images/models_NA/'


from machineLearning.config import PYTHON_PATH_USER_EMAIL_IMAGES,\
                                   tokenize

from machineLearning.config import censor_name,\
                                   load_censored_words,\
                                   reset__censored_words,\
                                   update_censored_words

#--- LOCAL PATHS FOR CREATING/DELETING FIGURES FOR PYTHON CONTROLLER
LOCALDIR = '/app'

PYTHON_PATH_CONFUSION_MATRIX = LOCALDIR + JS_PATH_CONFUSION_MATRIX
PYTHON_PATH_WORDCLOUD = LOCALDIR + JS_PATH_WORDCLOUD
PYTHON_PATH_PIES = LOCALDIR + JS_PATH_PIES
PYTHON_PATH_ROCS = LOCALDIR + JS_PATH_ROCS
PYTHON_PATH_MODELS = LOCALDIR + JS_PATH_MODELS



from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer


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


def create_confusion_figs(model,name_model, X_test, y_test, y_pred,extension=''):
    """
    Generates Confusion Matrix Figures
    """

    cnf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(4, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=True)

    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cnf_matrix, classes=['TAAK', 'NON_TAAK'], normalize=False)

    path = PYTHON_PATH_CONFUSION_MATRIX
    file_name = name_model+'/'+'cm' +str(extension)+ '.png'

    print('Saving  Confusion Matrix to '+path + file_name)
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()
    plt.close()
    full_path =  JS_PATH_CONFUSION_MATRIX+file_name

    return cnf_matrix, full_path


def create_wordcloud(X,y,model,name_model,extension=''):
    """
    Generates wordcloud
    """
    STOPS = set(stopwords.words('dutch'))
    DUTCHSTEMS = stemmer = DutchStemmer()

    Xselect = X[y == 0]
    text = ''
    for body in list(Xselect):
        text += ' '.join(tokenize(body,STOPS,DUTCHSTEMS))

    wordcloud = WordCloud(width=300, height=200).generate(text)
    fig = plt.figure(figsize=(3.8, 3.8))
    plt.imshow(wordcloud)
    plt.axis("off")
    path = PYTHON_PATH_WORDCLOUD
    file_name =  name_model+'/'+'wc'+str(extension)+ '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight')
    fig.clear()
    plt.close()
    full_path = JS_PATH_WORDCLOUD + file_name
    return full_path

def create_pie(name_model,ntrain, ntest,extension=''):
    """
    Generates Pie
    """
    labels = 'Training Set:\n ' + str(ntrain), 'Test Set:\n ' + str(ntest)
    sizes = [ntrain, ntest]
    colors = ['red', 'lightskyblue']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    fig = plt.figure(figsize=(3.6, 3.6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=45)

    path = PYTHON_PATH_PIES
    file_name = name_model+'/'+'pie'+extension+'.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

    full_path = JS_PATH_PIES + file_name
    return full_path



def create_ROC(name_model,cnf_matrix,y_test, y_pred,y_score,extension=''):
    """
    Generates ROC
    """

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
    path = PYTHON_PATH_ROCS
    file_name = name_model+'/'+'roc'+str(extension)+'.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()

    full_path = JS_PATH_ROCS + file_name
    return full_path

### model
def write_estimator(model,name_model,extension=''):
    """
    Writes off estimator
    """

    path = PYTHON_PATH_MODELS
    file_name = name_model+'/'+'model'+str(extension)+'.pkl'
    with open(path+file_name, 'wb') as fid:
        pickle.dump(model,fid)
    return path + file_name


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
    censored_list = load_censored_words()
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
            return censor_name(html_body,censored_list)


    return censor_name(html_body,censored_list)


def create_feature_importance_email(name_model, word_list, extra_info, user, top_n_words=15):
    """
    Generates feature importance for the email depending on the type of model:
    Multinomial Naive Bayes: rescaled plot of priors
    Trees: F-score
    """
    censored_list = load_censored_words()
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
        plt.xticks(np.arange(0, 3 * len(top_ham), 3) + 0.5, [shorten_word(censor_name(score[0],censored_list) ) for score in top_ham], rotation=45)
        plt.title('TAAK')
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(0, 3 * len(top_spam), 3), [min_score + score[1] for score in top_spam])
        plt.bar(np.arange(1, 3 * len(top_spam), 3), [min_score + score[2] for score in top_spam])
        plt.xticks(np.arange(0, 3 * len(top_spam), 3) + 0.5, [shorten_word(censor_name(score[0],censored_list) )  for score in top_spam], rotation=45)
        plt.title('NON_TAAK')

        plt.legend(['prob. ham', 'prob. spam'])

        path = PYTHON_PATH_USER_EMAIL_IMAGES  +user+'/feature_importance_email_NA/'+name_model + '/'

        file_name = 'efeature_imp_' + extra_info + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    elif (name_model == 'rf'):
        top_words = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        plt.bar(np.arange(0, len(top_words), 1), [score[1] for score in top_words])
        plt.xticks(np.arange(0, len(top_words), 1), [shorten_word(censor_name(score[0],censored_list) )  for score in top_words], rotation=45)

        path = PYTHON_PATH_USER_EMAIL_IMAGES  +user+'/feature_importance_email_NA/'+name_model + '/'
        file_name = 'efeature_imp_' + extra_info + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    elif (name_model == 'etr'):
        top_words = sorted(word_list, key=lambda x: (x[1]))[:-top_n_words - 1:-1]
        plt.bar(np.arange(0, len(top_words), 1), [score[1] for score in top_words])
        plt.xticks(np.arange(0, len(top_words), 1), [shorten_word(censor_name(score[0],censored_list) )  for score in top_words], rotation=45)

        path = PYTHON_PATH_USER_EMAIL_IMAGES +user+'/feature_importance_email_NA/'+name_model + '/'
        file_name = 'efeature_imp_' + extra_info + '.png'
        plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
        fig.clear()
        plt.close()
    else:
        pass


def create_prob_pie_email(name_model, score, extra_info, user,threshold=0.5):
    """
    Generates Pie of the score of the model for each email
    Score < threshold implies NON_TAAK Else TAAK
    """

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

    path = PYTHON_PATH_USER_EMAIL_IMAGES  +user+'/pie_probability_NA/'+name_model+'/'
    file_name = 'epie_prob_' +  extra_info + '.png'
    plt.savefig(path + file_name, transparent=True, bbox_inches='tight', orientation='portrait')
    fig.clear()
    plt.close()



