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
