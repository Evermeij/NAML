import psycopg2

import sqlite3

PATH_SQLITE = '/postgres/sqlite/database_NA_v1.db'
import os

PG_DB_NAME = 'naml'
PG_DB_USER = 'naml'
PG_DB_PASSWORD = 'naml'
PG_DB_HOST = 'db'


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
        sqlite_cursor.execute('''
        SELECT
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
        model_version
        FROM TABLE_MAILS WHERE (mail_id=?)
        ''', [(self.mail_id)])

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
    def insert_email_in_db_postgres(self, sqlite_cursor):
        email_data = (self.mail_id, str(self.date_sent), str(self.date_prediction), str(self.date_labelling),
                       str(self.date_conflict), self.from_email_address, self.to_email_address, self.subject,
                      (self.body.encode('ascii', errors='ignore')).decode('ascii'),
                       self.truth_class, self.labelling_user, self.pred_class, self.pred_score, self.is_labelled,
                       self.has_been_sent_back, self.is_in_conflict, self.model_hash, self.model_version)
        # insert mail if mail id is not present
        sqlite_cursor.execute('''
                                INSERT INTO TABLE_MAILS 
                                (
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
                                model_version
                                )
                                VALUES( %s,%s,%s,%s,%s,
                                        %s,%s,%s,%s,%s,
                                        %s,%s,%s,%s,%s,
                                        %s,%s,%s );
                                '''
                                  , email_data)
def openConnectionToPostgresDb():
    conn = psycopg2.connect("dbname={} user={} password={} host={}".format(PG_DB_NAME ,PG_DB_USER,PG_DB_PASSWORD,PG_DB_HOST))
    cur = conn.cursor()
    return conn, cur

def closeConnectionToPostgresDb(conn,cur):
    cur.close()
    conn.close()

def init_database():
    connPostgres, cursorPostgres = openConnectionToPostgresDb()
    connSqlite3 = sqlite3.connect(PATH_SQLITE )
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

if __name__ == '__main__':
    init_database()