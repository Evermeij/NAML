import os
import numpy as np
from flask import json

import re

LOCALDIR = '/app'

PYTHON_PATH_THRESHOLD = LOCALDIR + '/static/Images/'
PYTHON_FILENAME_THRESHOLD = 'thresholds.npy'


PYTHON_PATH_INFO_IMAGES = LOCALDIR + '/static/Images/'
PYTHON_FILENAME_INFO_IMAGES = 'filenames_imagesNA.npy'

PYTHON_PATH_USER_EMAIL_IMAGES = LOCALDIR+ '/static/Images/Emails/Users/'

PYTHON_PATH_JSON_INFO_EMAIL_IMAGES = LOCALDIR + '/static/Images/Emails/Users/'
PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES = 'json_email_data_NA.txt'

PYTHON_PATH_JSON_CENSOR = LOCALDIR + '/static/Images/Emails/'
PYTHON_FILENAME_JSON_CENSOR = 'json_censored.txt'

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


def load_filenames_images():
    filenames_dict = np.load(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES).item()
    return filenames_dict

def get_Email_names(user='Mette'):
    with open(PYTHON_PATH_JSON_INFO_EMAIL_IMAGES+user+'/'+PYTHON_FILENAME_JSON_INFO_EMAIL_IMAGES, 'r') as outfile:
        email_names = json.load(outfile)
    return email_names


def tokenize(text, stop, stemmer):
    """Converts text to tokens."""
    # tokens = word_tokenize(text, language='dutch')

    tokens = [word.lower() for word in text.split()]
    tokens = [i for i in tokens if i not in stop]
    tokens = ["".join(re.findall("[a-zA-Z]+", word)) for word in tokens]
    tokens = list(filter(lambda x: len(x) > 2, tokens))
    # tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def censor_name(text,list_censored_words):
    for name in list_censored_words:
        text = text.replace(name, '***')
    return text

def load_censored_words():
    with open(PYTHON_PATH_JSON_CENSOR + PYTHON_FILENAME_JSON_CENSOR, 'r') as outfile:
        list_censored =  json.load(outfile)
    return list_censored

def reset__censored_words():
    print('RESET CENSORED WORDS')
    clean_file(PYTHON_PATH_JSON_CENSOR,PYTHON_FILENAME_JSON_CENSOR)
    with open(PYTHON_PATH_JSON_CENSOR + PYTHON_FILENAME_JSON_CENSOR,'w') as outfile:
        json.dump([], outfile)

def update_censored_words(list_new_words):
    old_list_censored = load_censored_words()
    new_list = old_list_censored + list_new_words
    with open(PYTHON_PATH_JSON_CENSOR + PYTHON_FILENAME_JSON_CENSOR,'w') as outfile:
        json.dump(new_list, outfile)




