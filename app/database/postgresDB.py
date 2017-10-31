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
from machineLearning.config import censor_name,\
                                   load_censored_words,\
                                   reset__censored_words,\
                                   update_censored_words,\
                                   PYTHON_PATH_JSON_INFO_EMAIL_IMAGES

from flask import json

import os

PG_DB_NAME = 'naml'
PG_DB_USER = 'naml'
PG_DB_PASSWORD = 'naml'
PG_DB_HOST = 'db'
def openConnectionToPostgresDb():
    """
    Open connection to Postgress
    change hyperparameters here, i.e. password etc...
    """
    conn = psycopg2.connect("dbname={} user={} password={} host={}".format(PG_DB_NAME ,PG_DB_USER,PG_DB_PASSWORD,PG_DB_HOST))
    cur = conn.cursor()
    return conn, cur

def closeConnectionToPostgresDb(conn,cur):
    """
    Close connection to Postgress
    """
    cur.close()
    conn.close()

def load_current_ids(cursorPostgres):
    current_ids = []
    try:
        cursorPostgres.execute(''' SELECT DISTINCT
                         mail_id,date_sent
                         FROM TABLE_MAILS ''')
        for x in cursorPostgres:
            current_ids+=[x[0]]
    except:
        pass
    current_ids = list(set(current_ids))
    return current_ids

def init_database():
    connPostgres, cursorPostgres = openConnectionToPostgresDb()
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
    if (os.path.isfile(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE)):
            #--> initialize database with existing sqlite3
        connSqlite3 = sqlite3.connect(PYTHON_PATH_DATABASE + PYTHON_FILENAME_DATABASE )
        cursorSqlite3 = connSqlite3.cursor()

        #only initialize the table if table has not been created yet


        cursorSqlite3.execute(''' SELECT DISTINCT
                         mail_id, 
                         date_sent, 
                         from_email_address, 
                         to_email_address,
                         subject,
                         body,
                         truth_class,
                         is_labelled
                         FROM TABLE_MAILS ''')

        currentids = load_current_ids(cursorPostgres)


        for rowEmail in cursorSqlite3:
            if (rowEmail[0] not in currentids):
                currentids += [rowEmail[0]]
                email_obj = Email(mail_id=rowEmail[0], date_sent=rowEmail[1], date_prediction=None, date_labelling=None,
                               date_conflict=None, from_email_address=rowEmail[2], to_email_address= rowEmail[3], subject= rowEmail[4],
                               body= rowEmail[5], truth_class= rowEmail[6], labelling_user= None, pred_class= None, pred_score= None,
                               is_labelled= rowEmail[7], has_been_sent_back= False, is_in_conflict= False, model_hash= None,
                               model_version= None)
                try:
                    email_obj.insert_email_in_db_postgres(cursorPostgres)

                except Exception as e:
                    print('Could not transfer from sqlite3 to postgres...')
                    print(e)
                    print( email_obj.mail_id )
                    continue

        cursorSqlite3.close()
    else:
        currentids = load_current_ids(cursorPostgres)

        mail_id = '<welcome_to_datascience123456789>'

        if mail_id not in currentids:
            date_sent = ' 1990-05-01 00:00:00'
            from_email_address = 'benoitdescamps@hotmail.com'
            to_email_address = 'all@world.com'
            subject = 'Welkom \\ Welcome!'
            body = '''
            Dag Allemaal!\n
            
            Hartelijk bedankt,\n
            
            Geniet van de applicatie!\n
            
            Voor feedback, stuur gerust een email naar benoitdescamps@hotmail.com.\n
            
            groetjes,\n\n
            
            Benoit Descamps\n\n
            
            ......................................................................\n\n
            Dear All,\n
            thank you for your interest,\n
            Enjoy the application!\n
            Feedback is more than welcome, feel free to contact me at benoitdescamps@hotmail.com.\n
            
            regards,\n\n
            
            Benoit Descamps\n
            '''
            truth_class = None
            email_obj = Email(mail_id=mail_id, date_sent=date_sent, date_prediction=None, date_labelling=None,
                              date_conflict=None, from_email_address=from_email_address, to_email_address=to_email_address,
                              subject=subject,
                              body=body, truth_class=body , labelling_user=None, pred_class=None,
                              pred_score=None,
                              is_labelled=False, has_been_sent_back=False, is_in_conflict=False, model_hash=None,
                              model_version=None)
            email_obj.insert_email_in_db_postgres(cursorPostgres)


    connPostgres.commit()

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
    currentids = []
    for x in  cursorPostgres:
        if x[0] not in currentids: #this part is to make sure that there are not duplicates in the training/validation set
            currentids += [x[0]]
            df = pd.concat([df, pd.DataFrame({'Id': x[0], 'body': x[1], 'Target': x[2]}, index=[index_])])
            index_ += 1

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    df = df.loc[(df['body'].notnull()) & (df['Target'].notnull()), :]
    return df

# --- update prediction given some model
def get_new_predictions(ml_model,threshold=0.5):
    print('Start...getting new predictions for model...')
    update_dic = dict()
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    cursorPostgres.execute("SELECT mail_id,body FROM TABLE_MAILS;")
    for x in cursorPostgres:
        ypred = int(ml_model.predict_proba(np.array([x[1]]))[0, 0] < threshold)
        update_dic.update({str(x[0]): ypred})

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    return update_dic

def update_predictions(update_dic):
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    for id, pred in update_dic.items():
        cursorPostgres.execute("UPDATE TABLE_MAILS\
                    SET pred_class = %s\
                    WHERE mail_id = %s;", (str(pred), id))
    connPostgres.commit()
    closeConnectionToPostgresDb(connPostgres, cursorPostgres)

#def update_prediction_db(model_name='mbn'):


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
                 );''')
    currentids = load_current_ids(cursorPostgres)
    for filename_mail in os.listdir(path_mails):
        if filename_mail.split('.')[-1] == 'txt':
            try:
                print('Read file: ' + filename_mail)
                subject_, from_, to_, date_sent_, messageId_, body_ = extract_mail_properties(path_mails,
                                                                                              filename_mail)
            except:
                print('Could not process file')
                continue
            if messageId_ not in currentids:
                currentids+=[messageId_]
                email_obj = Email(mail_id=messageId_, date_sent=date_sent_, date_prediction=None, date_labelling=None,
                                  date_conflict=None, from_email_address=from_, to_email_address=to_, subject=subject_,
                                  body=body_, truth_class=None, labelling_user=None, pred_class=None, pred_score=None,
                                  is_labelled=0, has_been_sent_back=0, is_in_conflict=0, model_hash=None,
                                  model_version=None)
                email_obj.insert_email_in_db_postgres(cursorPostgres)

    # Save the changes when done
    print('Commit changes to database...')
    connPostgres.commit()
    # close connection
    print('Close connection to database...')
    closeConnectionToPostgresDb(connPostgres, cursorPostgres)

def get_mails_of(username, address, MAX_MAILS=30):
    censored_list = load_censored_words()
    current_ids = []
    n_mails = 0
    print('REFRESHING PAGE...')
    python_data = {"unknown_emails": []}

    print('Getting mails of ' + str(address))
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    if username == 'admin':
        cursorPostgres.execute(
            "SELECT DISTINCT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS WHERE is_labelled = %s Limit %s;",(0,int(MAX_MAILS),) ) #order by date_sent

    else:
        cursorPostgres.execute(
            "SELECT DISTINCT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS WHERE  is_labelled = %s AND (from_email_address LIKE %s OR to_email_address LIKE %s) Limit %s;",\
            (0,'%'+address+'%','%'+address+'%',int(MAX_MAILS),)) #order by date_sent

    for x in cursorPostgres:
        # if (x[8] != True):
            #if x[0] not in current_ids:
        current_ids+=[x[0]]
        if (len(x[2]) > 50):
            python_data["unknown_emails"] += [{
                'message_id': x[0],
                'class0': str(x[4]) == '0',#TAAK
                'class1': str(x[4]) == '1',# NON TAAK
                'header_from': censor_name(x[5],censored_list),
                'header_to': censor_name(x[6],censored_list),
                'header_subject': censor_name(x[2][:50] + ' ...',censored_list),
                'email_body': censor_name((x[3].encode('ascii', errors='ignore')).decode('ascii'),censored_list),
                'header_date': x[1],
                'is_corrected': 'black',  # white/blue
                'truth_class': x[7]
            }]
        else:
            python_data["unknown_emails"] += [{
                'message_id': x[0],
                'class0': str(x[4]) == '0',#TAAK
                'class1': str(x[4]) == '1',#NON TAAK
                'header_from': censor_name(x[5],censored_list),
                'header_to': censor_name(x[6],censored_list),
                'header_subject': censor_name(x[2],censored_list),
                'email_body': censor_name((x[3].encode('ascii', errors='ignore')).decode('ascii'),censored_list),
                'header_date': x[1],
                'is_corrected': 'black',  # white/blue
                'truth_class': x[7]
            }]
            n_mails += 1
        if n_mails > MAX_MAILS:
                break

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)
    return python_data


def load_user_mails(username,address, MAX_MAILS=30):
    '''
    initlize or loads json of emails_data of the user
    This way we do not have to reload the emails data everry time
    '''
    full_path_user_email_json = PYTHON_PATH_JSON_INFO_EMAIL_IMAGES + username + '/' + 'json_emails_' + username + '.txt'
    if (os.path.isfile(full_path_user_email_json)):
        with open(full_path_user_email_json, 'r') as outfile:
            email_data = json.load(outfile)
        return email_data
    else:
        email_data = get_mails_of(username, address, MAX_MAILS)
        with open(full_path_user_email_json, 'w') as outfile:
            json.dump(email_data, outfile)
        return email_data

def update_user_mails(username,address, MAX_MAILS=30):
    '''
   update json of emails_data of the user by reloading the mails
    This way we do not have to reload the emails data every time
    '''
    print('Upating emails data of user: '+str(username) )
    full_path_user_email_json = PYTHON_PATH_JSON_INFO_EMAIL_IMAGES + username + '/' + 'json_emails_' + username + '.txt'
    email_data = get_mails_of(username, address, MAX_MAILS)
    os.remove(full_path_user_email_json)
    with open(full_path_user_email_json, 'w') as outfile:
        json.dump(email_data, outfile)
    return email_data
def delete_user_mails(username):
    '''
    delete json of emails_data of the user by reloading the mails
    This way we do not have to reload the emails data every time
    '''
    print('Deleting emails data of user: '+str(username) )
    full_path_user_email_json = PYTHON_PATH_JSON_INFO_EMAIL_IMAGES + username + '/' + 'json_emails_' + username + '.txt'
    if (os.path.isfile(full_path_user_email_json)):
        os.remove(full_path_user_email_json)

def correct_predictions_from_input(mail_id, truth_class):
    connPostgres, cursorPostgres = openConnectionToPostgresDb()

    print('update truth class...')
    cursorPostgres.execute("UPDATE TABLE_MAILS\
                        SET truth_class = %s,is_labelled=%s \
                        WHERE mail_id = %s", ( str(truth_class), 1, str(mail_id) ))
    connPostgres.commit()

    closeConnectionToPostgresDb(connPostgres, cursorPostgres)

