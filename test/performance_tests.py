import pandas as pd
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import sys,os


path_database = os.getcwd()+'/../webapp/static/data/databases/'
filename_database = 'database_NA_v1.db'
#sys.path.append('webapp/')
sys.path.append(os.getcwd()+'/../machine_learning/')

import ml_model_v1 as ml


X_train, X_test, y_train, y_test = ml.get_train_test(path_database,filename_database,test_size=0.3)

name_model = 'etr'
estimator = ml.get_estimator(name_model)
ml.fit_model(X_train,y_train,estimator)
y_pred = ml.predict_target(X_test,name_model,estimator)
cnf_matrix = confusion_matrix(y_test, y_pred)
#    return cnf_matrix

def test_total_accuracy():
    accuracy = (cnf_matrix[0][0] + cnf_matrix[1][1]) / np.sum(cnf_matrix)
    assert(accuracy >= 0.7),'Accuracy below 0.7'
def test_taak_precision():
    accuracy = cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[0][1])
    assert(accuracy >= 0.7),'Accuracy below 0.7'
def test_nontaak_precision():
    accuracy = cnf_matrix[1][1]/(cnf_matrix[1][0]+cnf_matrix[1][1])
    assert(accuracy >= 0.7),'Accuracy below 0.7'