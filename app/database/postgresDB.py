'''
########################################################################################################################
                                            POSTGRES ADDITION
########################################################################################################################
'''
import psycopg2

import sqlite3
import pandas as pd

from datetime import datetime as dt
import numpy as np

from database.database import PYTHON_PATH_DATABASE,\
                            PYTHON_FILENAME_DATABASE
from database.Email import Email

import os

PG_DB_NAME = 'naml'
PG_DB_USER = 'naml'
PG_DB_PASSWORD = 'naml'
PG_DB_HOST = 'db'
def openConnectionToPostgresDb():
    conn = psycopg2.connect("dbname={} user={} password={} host={}".format(PG_DB_NAME ,PG_DB_USER,PG_DB_PASSWORD,PG_DB_HOST))
    cur = conn.cursor()
    return conn, cur

def closeConnectionToPostgresDb(conn,cur):
    cur.close()
    conn.close()

def init_database():
    connPostgres, cursorPostgres = openConnectionToPostgresDb()
    connSqlite3 = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE )
    cursorSqlite3 = connSqlite3.cursor()
    cursorPostgres.execute('''CREATE TABLE IF NOT EXISTS TABLE_MAILS
                    (
                 mail_id varchar, 
                 date_sent varchar, 
                 date_prediction varchar, 
                 date_labelling varchar, 
                 date_conflict varchar,
                 from_email_address varchar, 
                 to_email_address varchar,
                 subject varchar,
                 body varchar,
                 truth_class varchar,
                 labelling_user varchar,
                 pred_class varchar,
                 pred_score real,
                 is_labelled  smallint,
                 has_been_sent_back smallint,
                 is_in_conflict smallint,
                 model_hash varchar,
                 model_version varchar
                    )''')

    cursorSqlite3.execute(''' SELECT 
                     mail_id, 
                     date_sent, 
                     date_prediction, 
                     date_labelling, 
                     date_conflict,
                     from_email_address, 
                     to_email_address,
                     subject,
                     body,
                     truth_class,
                     labelling_user,
                     pred_class,
                     pred_score,
                     is_labelled,
                     has_been_sent_back,
                     is_in_conflict,
                     model_hash,
                     model_version FROM TABLE_MAILS ''')

    for rowEmail in cursorSqlite3:
        email_obj = Email(mail_id=rowEmail[0], date_sent=rowEmail[1], date_prediction=rowEmail[2], date_labelling=rowEmail[3],
                       date_conflict=rowEmail[4], from_email_address=rowEmail[5], to_email_address= rowEmail[6], subject= rowEmail[7],
                       body= rowEmail[8], truth_class= rowEmail[9], labelling_user= rowEmail[10], pred_class= rowEmail[11], pred_score= rowEmail[12],
                       is_labelled= rowEmail[13], has_been_sent_back= rowEmail[14], is_in_conflict= rowEmail[15], model_hash= rowEmail[16],
                       model_version= rowEmail[17])
        try:
            email_obj.insert_email_in_db_postgres(cursorPostgres)
        except:
            print('Could not transfer from sqlite3 to postgres...')
            print( email_obj.mail_id )
            continue


    connPostgres.commit()

    cursorSqlite3.close()
    connSqlite3.close()

    closeConnectionToPostgresDb(connPostgres, cursorPostgres )

'''
########################################################################################################################
                                            EMAIL DATABASE POSTGRES CODE
########################################################################################################################
'''

def get_mail(mail_id):

    df = pd.DataFrame()
    index_ = 0

    connPostgres, cursorPostgres = openConnectionToPostgresDb()
    cursorPostgres.execute(
        "SELECT mail_id,body,truth_class,date_sent,from_email_address,subject FROM TABLE_MAILS where mail_id= ANY(%s);",
        ([mail_id],) )
    for x in cursorPostgres:
        df = pd.concat([df, pd.DataFrame(
            {'Id': x[0], 'body': x[1], 'Target': x[2], 'Date': x[3], 'From': x[4], 'Subject': x[5]},
            index=[index_])])
        index_ += 1
        break

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    X = df['body'].astype(str).values

    target = df['Target'].astype(str).values[0]

    return X, target, df

def loadEmailsTrainData():
    df = pd.DataFrame()
    index_ = 0
    connPostgres, cursorPostgres = openConnectionToPostgresDb()
    cursorPostgres.execute("SELECT mail_id,body,truth_class FROM TABLE_MAILS;")

    for x in  cursorPostgres:
        df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2]}, index=[index_])])
        index_ += 1

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    df = df.loc[(df['body'].notnull()) & (df['Target'].notnull()), :]
    return df

# --- update prediction given some model
def get_new_predictions(ml_model):
    update_dic = dict()
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    cursorPostgres.execute("SELECT mail_id,body FROM TABLE_MAILS;")
    for x in cursorPostgres:
        ypred = ml_model.predict(np.array([x[1]]))
        update_dic.update({str(x[0]): ypred})

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    return update_dic

def update_predictions(update_dic):
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    for id, pred in update_dic.items():
        cursorPostgres.execute("UPDATE TABLE_MAILS\
                    SET pred_class = %s\
                    WHERE mail_id = %s", (str(pred[0]), id))
    connPostgres.commit()
    closeConnectionToPostgresDb(connPostgres, cursorPostgres)

'''
########################################################################################################################
                                            EMAIL DATABASE
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

def update_mail_database(path_mails):
    connPostgres, cursorPostgres = openConnectionToPostgresDb()
    # Create table
    cursorPostgres.execute('''CREATE TABLE IF NOT EXISTS TABLE_MAILS
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
                 );''')

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
            email_obj.insert_email_in_db(cursorPostgres)

    # Save the changes when done
    print('Commit changes to database...')
    connPostgres.commit()
    # close connection
    print('Close connection to database...')
    closeConnectionToPostgresDb(connPostgres, cursorPostgres)

def get_mails_of(username, address, MAX_MAILS=20):

    print('REFRESHING PAGE...')
    python_data = {"unknown_emails": []}

    print('Getting mails of ' + str(address))
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    if username == 'admin':
        cursorPostgres.execute(
            "SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS;")
    else:
        cursorPostgres.execute(
            "SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS WHERE to_email_address=%s OR from_email_address=%s ;",
            (address, address))
    n_mails = 0
    for x in cursorPostgres:
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

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    return python_data

def correct_predictions_from_input(mail_id, truth_class):
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    print('truth class  before update...')
    cursorPostgres.execute('SELECT mail_id, truth_class FROM TABLE_MAILS WHERE (mail_id=?)', [(mail_id)])
    entry = cursorPostgres.fetchone()
    print(entry)

    print('update truth class...')
    cursorPostgres.execute("UPDATE TABLE_MAILS\
                        SET truth_class = ?,is_labelled=? \
                        WHERE mail_id = ?", (truth_class, True, mail_id))
    connPostgres.commit()

    cursorPostgres.execute('SELECT mail_id, truth_class,is_labelled FROM TABLE_MAILS WHERE (mail_id=?)', [(mail_id)])
    entry = cursorPostgres.fetchone()
    print(entry)
    closeConnectionToPostgresDb(connPostgres, cursorPostgres)