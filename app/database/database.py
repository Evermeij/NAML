import pandas as pd
import sqlite3
import numpy as np
import os

from datetime import datetime as dt

LOCALDIR = '/app'
PYTHON_PATH_DATABASE = LOCALDIR + '/static/data/databases/'
PYTHON_FILENAME_DATABASE = 'database_NA_v1.db'

from database.Email import Email
'''
########################################################################################################################
                                            EMAIL DATABASE: SQLITE CODE
########################################################################################################################
'''

def get_string_between(start, end, email):
    if end == None:
        return email[email.find(start) + len(start)::]
    return email[email.find(start) + len(start):email.rfind(end)]

def extract_mail_properties(path, filename):
    try:
        with open(path + filename) as file:
            email = file.read()
    except:
        print('Could not process file')
        return None

    subject_ = get_string_between('__Subject__ :', '__From__ : ', email)
    from_ = get_string_between('__From__ :', '__To__ : ', email).replace('\n', '').replace(' ', '')
    to_ = get_string_between('__To__ :', '__Date__ : ', email).replace('\n', '').replace(' ', '')
    date_ = get_string_between('__Date__ :', '__MessageId__ : ', email).replace('\n', '')
    date_ = dt.strptime(date_, ' %Y-%m-%d %H:%M:%S')
    messageId_ = get_string_between('__MessageId__ :', '__Body__ : ', email).replace('\n', '').replace(' ', '')
    body_ = get_string_between('_Body__ :', None, email)
    return subject_, from_, to_, date_, messageId_, body_

def update_mail_database(path_database, filename_database, path_mails):
    conn = sqlite3.connect(path_database + filename_database)
    # create cursor
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS TABLE_MAILS
                 (
                 mail_id TEXT, 
                 date_sent TEXT, 
                 date_prediction TEXT, 
                 date_labelling TEXT, 
                 date_conflict TEXT,
                 from_email_address TEXT, 
                 to_email_address TEXT,
                 subject TEXT,
                 body TEXT,
                 truth_class TEXT,
                 labelling_user TEXT,
                 pred_class TEXT,
                 pred_score REAL,
                 is_labelled  BOOLEAN,
                 has_been_sent_back BOOLEAN,
                 is_in_conflict BOOLEAN,
                 model_hash TEXT,
                 model_version TEXT
                 )''')

    for filename_mail in os.listdir(path_mails):
        if filename_mail.split('.')[-1] == 'txt':
            try:
                print('Read file: ' + filename_mail)
                subject_, from_, to_, date_sent_, messageId_, body_ = extract_mail_properties(path_mails,
                                                                                              filename_mail)
            except:
                print('Could not process file')
                continue
            email_obj = Email(mail_id=messageId_, date_sent=date_sent_, date_prediction=None, date_labelling=None,
                              date_conflict=None, from_email_address=from_, to_email_address=to_, subject=subject_,
                              body=body_, truth_class=None, labelling_user=None, pred_class=None, pred_score=None,
                              is_labelled=False, has_been_sent_back=False, is_in_conflict=False, model_hash=None,
                              model_version=None)
            email_obj.insert_email_in_db(c)

    # Save the changes when done
    print('Commit changes to database...')
    conn.commit()
    # close connection
    print('Close connection to database...')
    conn.close()

def get_mails_of(username, address, MAX_MAILS=20):

    print('REFRESHING PAGE...')
    python_data = {"unknown_emails": []}

    print('Getting mails of ' + str(address))
    conn = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE)
    c = conn.cursor()
    if username == 'admin':
        c.execute(
            'SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS')
    else:
        c.execute(
            'SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS WHERE to_email_address=? OR from_email_address=? ',
            (address, address))
    n_mails = 0
    for x in c:
        if (x[8] != True):
            if (len(x[2]) > 50):
                python_data["unknown_emails"] += [{
                    'message_id': x[0],
                    'class0': x[4] == '0',
                    'class1': x[4] == '1',
                    'header_from': x[5],
                    'header_to': x[6],
                    'header_subject': x[2][:50] + ' ...',
                    'email_body': (x[3].encode('ascii', errors='ignore')).decode('ascii'),
                    'header_date': x[1],
                    'is_corrected': 'black',  # white/blue
                    'truth_class': x[7]
                }]
            else:
                python_data["unknown_emails"] += [{
                    'message_id': x[0],
                    'class0': x[4] == '0',
                    'class1': x[4] == '1',
                    'header_from': x[5],
                    'header_to': x[6],
                    'header_subject': x[2],
                    'email_body': (x[3].encode('ascii', errors='ignore')).decode('ascii'),
                    'header_date': x[1],
                    'is_corrected': 'black',  # white/blue
                    'truth_class': x[7]
                }]
            n_mails += 1
            if n_mails > MAX_MAILS:
                break

    conn.close()
    return python_data

def correct_predictions_from_input(mail_id, truth_class):
    conn = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE)
    c = conn.cursor()
    print('truth class  before update...')
    c.execute('SELECT mail_id, truth_class FROM TABLE_MAILS WHERE (mail_id=?)', [(mail_id)])
    entry = c.fetchone()
    print(entry)

    print('update truth class...')
    c.execute("UPDATE TABLE_MAILS\
                        SET truth_class = ?,is_labelled=? \
                        WHERE mail_id = ?", (truth_class, True, mail_id))
    conn.commit()

    c.execute('SELECT mail_id, truth_class,is_labelled FROM TABLE_MAILS WHERE (mail_id=?)', [(mail_id)])
    entry = c.fetchone()
    print(entry)
    conn.close()



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

def get_mail(mail_id):
    df = pd.DataFrame()
    index_ = 0
    conn = sqlite3.connect(PYTHON_PATH_DATABASE +PYTHON_FILENAME_DATABASE)
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

def loadEmailsTrainData():
    df = pd.DataFrame()
    index_ = 0
    conn = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE )
    c = conn.cursor()
    c.execute('SELECT mail_id,body,truth_class FROM TABLE_MAILS ')
    for x in c:
        df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2]}, index=[index_])])
        index_ += 1
    conn.close()
    df = df.loc[(df['body'].notnull()) & (df['Target'].notnull()), :]
    return df

#--- update prediction given some model
def get_new_predictions(ml_model):
    update_dic = dict()
    conn = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE)
    c = conn.cursor()
    c.execute('SELECT mail_id,body FROM TABLE_MAILS ')
    for x in c:
        ypred = ml_model.predict(np.array([x[1]]))
        update_dic.update({str(x[0]):ypred})
    conn.close()
    return update_dic

def update_predictions(update_dic):
    conn = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE)
    c = conn.cursor()
    for id,pred in update_dic.items():
        c.execute("UPDATE TABLE_MAILS\
                    SET pred_class = ?\
                    WHERE mail_id = ?",(str(pred[0]),id))
    conn.commit()
    conn.close()
