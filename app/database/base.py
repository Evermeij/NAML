from abc import ABC, abstractmethod


class BaseEmail(ABC):
    '''
    Base Class of for the Emails
    '''
    @abstractmethod
    def insert_email_in_db(self, sqlite_cursor):
        '''
        Initiates the sqlite database:
        -> will not be used anymore
        '''
        pass

    @abstractmethod
    def insert_email_in_db_postgres(self, sqlite_cursor):
        '''
        Initiates the sqlite postgres database:
        '''
        pass
