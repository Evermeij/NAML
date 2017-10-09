from flask import json

'''
MACHINE LEARNING
'''


from flask import json
import os
import sqlite3 #for production go over to postgresql

import numpy as np

from datetime import datetime as dt
import sys,os

path_database = os.getcwd()+'/static/data/databases/'
filename_database = 'database_NA_v1.db'
path_mails = os.getcwd()+'/static/data/processed'

path_model = os.getcwd()+'/static/models_NA/'
filename_model = 'model_1980_1386_594_1_1.pkl'

path_corrections = os.getcwd()+'/static/data/corrections/'
filename_corrections = 'corr.txt'

'''
DATABASE PIPELINE
'''


'''
########################################################################################################################
                                            EMAIL DATABASE
#######################################################################################################################
'''
def get_string_between(start,end,email):
    if end == None:
        return email[email.find(start)+len(start)::]
    return email[email.find(start)+len(start):email.rfind(end)]
def extract_mail_properties(path,filename):
    try:
        with open(path+filename) as file:
            email = file.read()
    except:
        print('Could not process file')
        return None

    subject_ = get_string_between('__Subject__ :','__From__ : ',email)
    from_ = get_string_between('__From__ :','__To__ : ',email).replace('\n','').replace(' ','')
    to_ = get_string_between('__To__ :','__Date__ : ',email).replace('\n','').replace(' ','')
    date_ = get_string_between('__Date__ :','__MessageId__ : ',email).replace('\n','')
    date_ = dt.strptime(date_, ' %Y-%m-%d %H:%M:%S')
    messageId_ = get_string_between('__MessageId__ :','__Body__ : ',email).replace('\n','').replace(' ','')
    body_ = get_string_between('_Body__ :',None,email)
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


def get_mails_of(username,address,MAX_MAILS = 20):

    print('REFRESHING PAGE...')
    python_data = {"unknown_emails":[]}

    print('Getting mails of '+str(address) )
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    if username=='admin':
        c.execute('SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS')
    else:
        c.execute('SELECT mail_id,date_sent,subject,body,pred_class,from_email_address,to_email_address, truth_class, is_labelled FROM TABLE_MAILS WHERE to_email_address=? OR from_email_address=? ',(address,address) )
    n_mails = 0
    for x in c:
        if (x[8]!= True):
            if (len(x[2])>50):
                python_data["unknown_emails"]+=[{
                    'message_id': x[0],
                    'class0': x[4]=='0',
                    'class1': x[4]=='1',
                    'header_from': x[5],
                    'header_to': x[6],
                    'header_subject': x[2][:50]+' ...',
                    'email_body': (x[3].encode('ascii', errors='ignore')).decode('ascii'),
                    'header_date': x[1],
                    'is_corrected': 'black', #white/blue
                    'truth_class' : x[7]
                    }]
            else:
                python_data["unknown_emails"]+=[{
                    'message_id': x[0],
                    'class0': x[4]=='0',
                    'class1': x[4]=='1',
                    'header_from': x[5],
                    'header_to': x[6],
                    'header_subject': x[2],
                    'email_body': (x[3].encode('ascii', errors='ignore')).decode('ascii'),
                    'header_date': x[1],
                    'is_corrected': 'black', #white/blue
                    'truth_class' : x[7]
                    }]
            n_mails+=1
            if n_mails> MAX_MAILS:
                break

    conn.close()
    return python_data

def correct_predictions_from_input(mail_id,truth_class):
    conn = sqlite3.connect(path_database + filename_database)
    c = conn.cursor()
    print('truth class  before update...' )
    c.execute('SELECT mail_id, truth_class FROM TABLE_MAILS WHERE (mail_id=?)', [(mail_id)])
    entry = c.fetchone()
    print(entry)

    print('update truth class...')
    c.execute("UPDATE TABLE_MAILS\
                        SET truth_class = ?,is_labelled=? \
                        WHERE mail_id = ?", (truth_class,True, mail_id) )
    conn.commit()

    c.execute('SELECT mail_id, truth_class,is_labelled FROM TABLE_MAILS WHERE (mail_id=?)', [(mail_id)])
    entry = c.fetchone()
    print(entry)
    conn.close()

class Email():
    def __init__(self, mail_id, date_sent, date_prediction, date_labelling, date_conflict, from_email_address,
                 to_email_address, subject, body, truth_class, labelling_user, pred_class, pred_score, is_labelled,
                 has_been_sent_back, is_in_conflict, model_hash, model_version):
        """
        email being classified into different classes which need to be confirmed by the user
        """
        self.mail_id = mail_id
        self.date_sent = date_sent
        self.date_prediction = date_prediction
        self.date_labelling = date_labelling
        self.date_conflict = date_conflict
        self.from_email_address = from_email_address
        self.to_email_address = to_email_address
        self.subject = subject
        self.body = body
        self.truth_class = truth_class
        self.labelling_user = labelling_user
        self.pred_class = pred_class
        self.pred_score = pred_score
        self.is_labelled = is_labelled
        self.has_been_sent_back = has_been_sent_back
        self.is_in_conflict = is_in_conflict
        self.model_hash = model_hash
        self.model_version = model_version

    def insert_email_in_db(self, sqlite_cursor):
        email_data = [(self.mail_id, str(self.date_sent), str(self.date_prediction), str(self.date_labelling),
                       str(self.date_conflict), self.from_email_address, self.to_email_address, self.subject, self.body,
                       self.truth_class, self.labelling_user, self.pred_class, self.pred_score, self.is_labelled,
                       self.has_been_sent_back, self.is_in_conflict, self.model_hash, self.model_version)]
        # insert mail if mail id is not present
        sqlite_cursor.execute('SELECT * FROM TABLE_MAILS WHERE (mail_id=?)', [(self.mail_id)])
        entry = sqlite_cursor.fetchone()
        if entry == None:
            print('new entry added...')
            sqlite_cursor.executemany("INSERT INTO TABLE_MAILS VALUES(?,?,?,?,?,\
                                                              ?,?,?,?,?,\
                                                              ?,?,?,?,?,\
                                                              ?,?,?)", email_data)
        else:
            print('email is already in database and has not been added..')

            # user data? user_id? date of confirmed
            # def predict(model):

'''
########################################################################################################################
                                            CONFLICT DATABASE
########################################################################################################################
'''
def update_conflicts_database(path_database,filename_database,mail_id,date_conflict,previous_labelling_user,\
                              conflicting_user,previous_label,conflicting_label):
    conn = sqlite3.connect(path_database + filename_database)
    #create cursor
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS TABLE_CONFLICTS
                 (
                 mail_id TEXT,
                 date_conflict TEXT,
                 previous_labelling_user TEXT,
                 conflicting_user TEXT,
                 previous_label INTEGER,
                 conflicting_label INTEGER
                 )''')
    print('new conflict added...')
    c.executemany("INSERT INTO TABLE_CONFLICTS VALUES(?,?,?,?,?,?)",\
            [(mail_id,date_conflict,previous_labelling_user,conflicting_user,previous_label,conflicting_label)])
    # Save the changes when done
    print('Commit changes to database...')
    conn.commit()
    #close connection
    print('Close connection to database...')
    conn.close()

'''
########################################################################################################################
                                                    MACHINE LEARNING
########################################################################################################################
'''

class Model():
    def __init__(self, model, model_hash, model_version, n_classes):
        # load model from pickle or create new pickle
        self.model = model

        # load latest data hash and model_version
        self.model_hash = model_hash
        self.model_version = model_version
        self.model_description = ''
        self.n_classes = n_classes

    def add_class_columns(self, path_database, filename_database):
        # Hence check number 'number_of_mails_predicted_class_...' = n_classes

        conn = sqlite3.connect(path_database + filename_database)
        # create cursor
        c = conn.cursor()
        c.execute('SELECT * FROM TABLE_MODEL_PROPERTIES')
        columns = [x[0] for x in c.description]
        training_class_cols = list(filter(lambda x: 'training_number_of_mails_class_' in x, columns))
        if (len(training_class_cols) <= self.n_classes):
            for n in range(len(training_class_cols), self.n_classes):
                print('add new column...' + str(n))
                c.execute("alter table TABLE_MODEL_PROPERTIES \
                add column %s INTEGER " % ('number_of_mails_predicted_class_' + str(n)))
                c.execute("alter table TABLE_MODEL_PROPERTIES \
                add column %s INTEGER " % ('number_of_mails_labelled_class_' + str(n)))
                c.execute("alter table TABLE_MODEL_PROPERTIES \
                add column %s INTEGER " % ('training_number_of_mails_class_' + str(n)))
                c.execute("alter table TABLE_MODEL_PROPERTIES \
                add column %s INTEGER " % ('eval_number_of_mails_class_' + str(n)))
        else:
            print('No columns were added...')
        conn.close()

    def add_metric_columns(self, path_database, filename_database, metrics_values):
        # Hence check whether metric columns match

        conn = sqlite3.connect(path_database + filename_database)
        # create cursor
        c = conn.cursor()
        c.execute('SELECT * FROM TABLE_MODEL_PROPERTIES')
        columns = [x[0] for x in c.description]
        metric_cols = list(filter(lambda x: 'evaluation_metric_description_' in x, columns))
        if (len(metric_cols) <= len(metrics_values)):
            for n in range(len(metric_cols), len(metrics_values)):
                print('add new column...' + str(n))
                c.execute("alter table TABLE_MODEL_PROPERTIES \
                add column %s 'TEXT' " % ('evaluation_metric_description_' + str(n)))
                c.execute("alter table TABLE_MODEL_PROPERTIES \
                add column %s REAL " % ('evaluation_metric_value_' + str(n)))
        else:
            print('No columns were added...')
        conn.close()

    def update_model_database(self, path_database, filename_database, X, y, y_pred, X_train, y_train, X_eval, y_eval,
                              metrics_descriptions, metrics_values):
        conn = sqlite3.connect(path_database + filename_database)
        # create cursor
        c = conn.cursor()
        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS TABLE_MODEL_PROPERTIES
                 (
                 date_update TEXT, 
                 model_hash TEXT,
                 model_version INTEGER, 
                 model_description TEXT, 
                 number_of_mails_predicted_class_0 INTEGER, 
                 number_of_mails_predicted_class_1 INTEGER,
                 number_of_mails_labelled_class_0 INTEGER, 
                 number_of_mails_labelled_class_1 INTEGER,
                 size_training_set INTEGER,
                 training_number_of_mails_class_0 INTEGER, 
                 training_number_of_mails_class_1 INTEGER,
                 size_evaluation_set INTEGER,
                 eval_number_of_mails_class_0 INTEGER, 
                 eval_number_of_mails_class_1 INTEGER,
                 evaluation_metric_description_0 TEXT,
                 evaluation_metric_value_0 REAL 
                 )''')
        conn.close()
        date_update = str(dt.now())
        model_hash = self.model_hash
        model_version = self.model_version
        model_description = self.model_description
        size_training_set = X_train.shape[0]
        size_evaluation_set = X_eval.shape[0]

        columns = 'date_update, model_hash, model_version,  model_description, size_training_set, size_evaluation_set'
        new_row = [date_update, model_hash, model_version, model_description, size_training_set, size_evaluation_set]
        values_string_code = '(?,?,?,?,?,?'

        # update columns classes
        self.add_class_columns(path_database, filename_database)
        # update columns metrics
        self.add_metric_columns(path_database, filename_database, metrics_values)

        for n in range(self.n_classes):
            columns += ', ' + 'number_of_mails_predicted_class_' + str(n)
            new_row += [int(np.sum(y_pred == n))]
            values_string_code += ',?'

            columns += ', ' + 'number_of_mails_labelled_class_' + str(n)
            new_row += [int(np.sum(y == n))]
            values_string_code += ',?'

            columns += ', ' + 'training_number_of_mails_class_' + str(n)
            new_row += [int(np.sum(y_train == n))]
            values_string_code += ',?'

            columns += ', ' + 'eval_number_of_mails_class_' + str(n)
            new_row += [int(np.sum(y_eval == n))]
            values_string_code += ',?'

        for n in range(len(metrics_values)):
            columns += ', ' + 'evaluation_metric_description_' + str(n)
            new_row += [metrics_descriptions[n]]
            values_string_code += ',?'

            columns += ', ' + 'evaluation_metric_value_' + str(n)
            new_row += [metrics_values[n]]
            values_string_code += ',?'
        values_string_code = values_string_code + ')'

        # insert new row
        conn = sqlite3.connect(path_database + filename_database)
        c = conn.cursor()
        c.execute("SELECT %s FROM TABLE_MODEL_PROPERTIES" % (columns))
        c.executemany("INSERT INTO TABLE_MODEL_PROPERTIES VALUES %s" % (values_string_code), [tuple(new_row)])
        # Save the changes when done
        print('Commit changes to database...')
        conn.commit()
        # close connection
        print('Close connection to database...')
        conn.close()

'''
########################################################################################################################
                                                    SCORES DATABASE
########################################################################################################################
'''
def update_scores_database(path_database,filename_database,mail_id,model_hash,date_update_score,pred_score,pred_class,
                           truth_class):
    conn = sqlite3.connect(path_database + filename_database)
    #create cursor
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS TABLE_SCORES
                 (
                 mail_id TEXT,
                 model_hash TEXT,
                 date_update_score TEXT,
                 pred_score REAL,
                 pred_class TEXT,
                 truth_class TEXT )''')
    print('add score...')
    c.executemany("INSERT INTO TABLE_SCORES VALUES(?,?,?,?,?,?)",\
        [(mail_id,model_hash,date_update_score,pred_score,pred_class,truth_class)])
    # Save the changes when done
    print('Commit changes to database...')
    conn.commit()
    #close connection
    print('Close connection to database...')
    conn.close()


'''
VISUALS
'''

def get_Image_names():
    print('I am here...trying to load images from static/Images/json_filenames_NA.txt...')
    print(os.getcwd())
    with open('static/Images/json_filenames_NA.txt', 'r') as outfile:
        filenames = json.load(outfile)
    return filenames

def get_Email_names(user='Mette'):
    with open('static/Images/Emails/Users/'+user+'/json_email_data_NA.txt', 'r') as outfile:
        email_names = json.load(outfile)
    return email_names
'''
RETRAIN MODEL
'''

python_data = {"unknown_emails":[{
                            'message_id': 1,
                            'class0':True,
                            'class1':False,
                            'header_from': "Kimberly Grant",
                            'header_to': "Lize Lewis",
                            'header_subject': "subject 1",
                            'email_body': "Hi, how are you? Grant here...",
                            'header_date':"17-01-2017"
                        }, {
                            'message_id': 2,
                            'class0':False,
                            'class1':True,
                            'header_from': "Elizabeth Lewis",
                            'header_to': "Kim Grant",
                            'header_subject': "subject 2",
                            'email_body': "Hi, how are you? Liz here...",
                            'header_date':"25-12-2016"
                        },{
                            'message_id': 3,
                            'class0':False,
                            'class1':True,
                            'header_from': "Shawn Ellis",
                            'header_to': "Lize Lewis",
                            'header_subject': "subject 3",
                            'email_body': "Hi, how are you? Shawn here...",
                            'header_date':"16-10-2016"
                        },{
                            'message_id': 4,
                            'class0':True,
                            'class1':False,
                            'header_from': "Shawn Ellis",
                            'header_to': "Kim Grant",
                            'header_subject': "subject 4",
                            'email_body': "Hi, how are you? Sh here...",
                            'header_date':"13-09-2016"
                        },{
                            'message_id': 5,
                            'class0':False,
                            'class1':True,
                            'header_from': "Shawn Ellis",
                            'header_to': "Lize G.",
                            'header_subject': "subject 5",
                            'email_body': "Hi, how are you? I am here...",
                            'header_date':"05-07-2016"
                        },{
                            'message_id': 6,
                            'class0':False,
                            'class1':True,
                            'header_from': "Shawn Ellis",
                            'header_to': "Lize T.",
                            'header_subject': "subject 5",
                            'email_body': "Hi, how are you? Ellis here...",
                            'header_date':"12-02-2016"
                        }
]}
