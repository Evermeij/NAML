from abc import ABC, abstractmethod


class BaseEmail(ABC):
    '''
    Base Class of the Machine Learning Project
    '''
    @abstractmethod
    def insert_email_in_db(self, sqlite_cursor):
       pass

    @abstractmethod
    def insert_email_in_db_postgres(self, sqlite_cursor):
        pass
