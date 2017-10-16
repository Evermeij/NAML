from abc import ABC, abstractmethod


class BaseMLProject(ABC):
    '''
    Base Class of the Machine Learning Project
    '''
    @abstractmethod
    def get_estimator(self,model_name):
       pass

    @abstractmethod
    def get_train_test(self,path_database, filename_database, test_size):
        pass

    @abstractmethod
    def fit_model(self,X, y, estimator, weights):
        pass

    @abstractmethod
    def predict_target(self,X, name_model, estimator, threshold):
        pass

    @abstractmethod
    def fit_grid_search(self,X, y, name_model, n_splits, n_iter):
        pass

    @abstractmethod
    def get_logProb(self,estimator, name_model, class_label):
       pass

    @abstractmethod
    def get_model_properties(self,estimator, name_model, class_label):
        pass

    @abstractmethod
    def loadBestEstimator(self,name_model):
        pass

    @abstractmethod
    def create_word_list(self, X_transformed, estimator, name_model, key_words):
        pass

    @abstractmethod
    def getJsonEmailData(self, user):
        pass

    @abstractmethod
    def deleteJsonEmailData(self, user):
        pass

    @abstractmethod
    def saveJsonEmailData(self, user, json_email_data):
        pass