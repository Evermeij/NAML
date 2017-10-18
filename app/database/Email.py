from database.base import BaseEmail

class Email(BaseEmail):
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
                       self.truth_class, self.labelling_user, self.pred_class, self.pred_score, int(self.is_labelled),
                       int(self.has_been_sent_back), int(self.is_in_conflict), self.model_hash, self.model_version)
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