import os
import pickle
import re
import sqlite3

import numpy as np

import sys



path_database = os.getcwd()+'/../webapp/static/data/databases/'
filename_database = 'database_NA_v1.db'
path_mails = os.getcwd()+'/../webapp/static/data/processed/'

path_model = 'static/models_NA/'
filename_model = 'model_1980_1386_594_1_1.pkl'

path_corrections = 'static/data/corrections/'
filename_corrections = 'corr.txt'

sys.path.append(os.getcwd()+'/../machine_learning/')
sys.path.append(os.getcwd()+'/../webapp/')
import ml_model as ml
import model

estimator = ml.estimator

def update_data():
    print('UPDATE DATABASE...')
    model.update_mail_database(path_database, filename_database, path_mails)

def get_mails_of(address):
    python_data = {"unknown_emails":[]}
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address FROM TABLE_MAILS WHERE to_email_address=? OR from_email_address=? ',(address,address) )
    for x in c:
        python_data["unknown_emails"]+=[{
                        'message_id': x[0],
                        'class0': x[4]=='0',
                        'class1': x[4]=='1',
                        'header_from': x[5],
                        'header_to': x[6],
                        'header_subject': x[2],
                        'email_body': x[3],
                        'header_date': x[1],
                        'is_corrected': 'black' #white/blue
                        }]
    conn.close()
    return python_data

def get_new_predictions(path_model, filename_model,path_database, filename_database):
    update_dic = dict()
    with open(path_model+ filename_model, 'rb') as fid:
        ml_model = pickle.load(fid)
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

def correct_predictions_from_file(path_corrections,filename_corrections, path_database, filename_database):
    print('reading')
    with open(path_corrections + filename_corrections, 'r') as file:
        print(file.read())
        corrections = file.read()
        print(corrections)
    #remove file after extracting all corrections
    os.remove(path_corrections + filename_corrections)
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    print(corrections)
    for correction in corrections:
        correction = correction.split(';')
        print(correction)
        print('correct id : '+correction[0].replace(' ','') )
        c.execute("UPDATE TABLE_MAILS\
                        SET truth_class = ?\
                        WHERE mail_id = ?", (correction[1].replace(' ',''), correction[0].replace(' ','')) )
    conn.commit()
    conn.close()

def correct_predictions_from_input(mail_id,truth_class):
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    c.execute("UPDATE TABLE_MAILS\
                        SET truth_class = ?\
                        WHERE mail_id = ?", (truth_class, mail_id) )
    conn.commit()
    conn.close()


if __name__ == '__main__':
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    c.execute('SELECT mail_id,truth_class FROM TABLE_MAILS WHERE mail_id = ?',(["<5535528019EF6A4FAE0EBBEB9889DE2917872419@OWWNLMBX101.ocw.local>"] ))
    n = 0
    for x in c:
        print(x)
    conn.close()
