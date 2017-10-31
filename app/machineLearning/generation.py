import numpy as np

import os

from flask import json

#from database.database import get_mail,update_predictions,get_new_predictions
from database.postgresDB import get_mail,update_predictions,get_new_predictions


from visualisations.visualisation import PYTHON_PATH_CONFUSION_MATRIX , \
                                           PYTHON_PATH_WORDCLOUD, \
                                           PYTHON_PATH_PIES, \
                                           PYTHON_PATH_ROCS, \
                                           PYTHON_PATH_MODELS,\
                                           create_confusion_figs,\
                                            return_html_body,\
                                            create_ROC,\
                                            create_pie,\
                                            create_wordcloud,\
                                            write_estimator,\
                                            create_prob_pie_email,\
                                            create_feature_importance_email
from machineLearning.config import PYTHON_PATH_INFO_IMAGES,\
                    PYTHON_FILENAME_INFO_IMAGES,\
                    clean_dir,\
                    clean_file,\
                    load_filenames_images

from machineLearning.config import censor_name,\
                                   load_censored_words,\
                                   reset__censored_words,\
                                   update_censored_words

from machineLearning.ml_model import MLProject

def generate_new_images_manual_fit(name_model ='mnb',thres = 0.5,weight_taak = 0.5,weight_non_taak =0.49):
    print('Clean Confusion Matrices...')
    clean_dir( PYTHON_PATH_CONFUSION_MATRIX ,extra_dir=name_model+'/' )
    print('Clean Wordcloud...')
    clean_dir(PYTHON_PATH_WORDCLOUD,extra_dir=name_model+'/')
    print('Clean Pie...')
    clean_dir(PYTHON_PATH_PIES,extra_dir=name_model+'/')
    print('Clean Roc...')
    clean_dir(PYTHON_PATH_ROCS,extra_dir=name_model+'/')
    print('Clean Models...')
    clean_dir(PYTHON_PATH_MODELS,extra_dir=name_model+'/')

    #--- start new project
    newProject = MLProject()
    threshold_dic = newProject.get_threshold_dic()
    print('update Threshold dictionary')
    newProject.set_threshold_dic(name_model, thres)

    estimator = newProject.get_estimator(name_model)
    X_train, X_test, y_train, y_test = newProject.get_train_test(test_size=0.33)
    print('Start fit model...')
    estimator = newProject.fit_model(X_train, y_train, estimator,weights=[weight_taak,weight_non_taak])

    ntrain = len(X_train)
    ntest = len(X_test)

    y_score = estimator.predict_proba(X_test)
    y_pred = (y_score[:, 0] < thres).astype(int)

    print('Generating new Figures...')
    new_image_extension = str(np.random.rand()).replace('.', '')
    print('Generating new confusion matrix...')
    cnf_matrix,filename_cm= create_confusion_figs(estimator, name_model, X_test, y_test, y_pred, extension=new_image_extension)
    print('Generating new roc...')
    filename_roc = create_ROC(name_model, cnf_matrix, y_test, y_pred, y_score,extension=new_image_extension)
    print('Generating new pie...')
    filename_pie = create_pie(name_model,ntrain, ntest,extension=new_image_extension)
    print('Generating new wordcloud...')
    filename_wc =create_wordcloud(X_train,y_train, estimator, name_model,extension=new_image_extension)
    print('Writing off estimator...')
    filename_model =write_estimator(estimator, name_model,extension=new_image_extension)

    print('Updating filenames...')
    filenames_dict = np.load(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES).item()
    filenames_dict[name_model]['pie'] = filename_pie
    filenames_dict[name_model]['wc'] = filename_wc
    filenames_dict[name_model]['roc'] = filename_roc
    filenames_dict[name_model]['cm'] = filename_cm
    #filenames_dict[name_model]['model'] = filename_model

    os.remove(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES)
    np.save(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES, filenames_dict)
    return filenames_dict


def generate_new_images_auto_fit(name_model ='mnb'):
    print('Clean Confusion Matrices...')
    clean_dir( PYTHON_PATH_CONFUSION_MATRIX, extra_dir=name_model+'/' )
    print('Clean Wordcloud...')
    clean_dir(PYTHON_PATH_WORDCLOUD,extra_dir=name_model+'/')
    print('Clean Pie...')
    clean_dir(PYTHON_PATH_PIES,extra_dir=name_model+'/')
    print('Clean Roc...')
    clean_dir(PYTHON_PATH_ROCS,extra_dir=name_model+'/')
    print('Clean Models...')
    clean_dir(PYTHON_PATH_MODELS,extra_dir=name_model+'/')

    newProject = MLProject()
    X_train, X_test, y_train, y_test = newProject.get_train_test(test_size=0.33)

    result_grid_search = newProject.fit_grid_search(X_train,y_train,name_model,n_splits=3,n_iter=10)
    estimator = result_grid_search['opt_estimator']

    newProject.set_threshold_dic(name_model, result_grid_search['opt_thres'])
    threshold_dic = newProject.get_threshold_dic()

    ntrain = len(X_train)
    ntest = len(X_test)

    y_score = estimator.predict_proba(X_test)
    y_pred = (y_score[:, 0] < threshold_dic[name_model]).astype(int)

    new_image_extension = str(np.random.rand()).replace('.', '')
    cnf_matrix,filename_cm= create_confusion_figs(estimator, name_model, X_test, y_test, y_pred, extension=new_image_extension)
    filename_roc = create_ROC(name_model, cnf_matrix, y_test, y_pred, y_score,extension=new_image_extension)
    filename_pie = create_pie(name_model,ntrain, ntest,extension=new_image_extension)
    filename_wc =create_wordcloud(X_train,y_train, estimator, name_model,extension=new_image_extension)
    filename_model =write_estimator(estimator, name_model,extension=new_image_extension)

    filenames_dict = np.load(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES).item()
    filenames_dict[name_model]['pie'] = filename_pie
    filenames_dict[name_model]['wc'] = filename_wc
    filenames_dict[name_model]['roc'] = filename_roc
    filenames_dict[name_model]['cm'] = filename_cm
    #filenames_dict[name_model]['model'] = filename_model

    os.remove(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES)
    np.save(PYTHON_PATH_INFO_IMAGES+PYTHON_FILENAME_INFO_IMAGES, filenames_dict)


    print('Update predictions...')
    update_dic = get_new_predictions(estimator,threshold=threshold_dic[name_model])
    update_predictions(update_dic)
    print('Predictions updated!')

    return filenames_dict


def add_new_email_images(mail_id,user='Mette'):
    censored_list = load_censored_words()

    currentProject = MLProject() #--- this part could be improved

    threshold_dic = currentProject.get_threshold_dic()
    spam_ham_dic = {'0': 'TAAK', '1': 'NON_TAAK'}
    def shorten_word(word,MAX_LEN=35):
        if len(word)>MAX_LEN:
            return word[:MAX_LEN]+'...'
        return word

    json_email_data = currentProject.getJsonEmailData(user)

    X,target,df =  get_mail(mail_id)



    for name_model in currentProject.MODEL_DICTIONARY.keys():
        estimator = currentProject.loadBestEstimator(name_model)
        log_probs, words_key, key_words = currentProject.get_model_properties(estimator, name_model, 'TAAK')



        body = X
        date = df['Date']
        _from = df['From']
        subject = df['Subject']
        X_transformed = estimator.named_steps['vectorizer'].transform(body)

        word_list = currentProject.create_word_list(X_transformed, estimator, name_model, key_words)

        score = estimator.predict_proba(body)
        y_pred = int(score[0][0] < threshold_dic[name_model])


        print(X_transformed.shape)

        html_body = return_html_body(body[0], word_list, y_pred, top_n_words=20)
        extra_info = 'email_' + mail_id.replace('.','').replace('>','').replace('<','').replace('/','').replace('\\','')
        create_prob_pie_email(name_model, score[0][0], extra_info , user,threshold_dic[name_model])
        create_feature_importance_email(name_model, word_list, extra_info ,user, top_n_words=5)

        print('here...')
        #print(y)
        email_data = {'pred': spam_ham_dic[str(y_pred)],
                        'truth': spam_ham_dic.get(target,'NONE'),
                        'date': date[0],
                        'from': censor_name(_from[0],censored_list),
                        'subject': shorten_word(censor_name(subject[0],censored_list)),
                        'html_body': html_body,
                        'eFimp': "/static/Images/Emails/Users/"+user+'/feature_importance_email_NA/' + name_model + '/' + "efeature_imp_" + extra_info + '.png',
                        'epie': "/static/Images/Emails/Users/" +user+'/pie_probability_NA/'+ name_model + '/' + "epie_prob_" + extra_info + '.png'}
        if name_model not in json_email_data.keys():
            json_email_data[name_model] = list([email_data])
        else:
            json_email_data[name_model]+= [email_data]

    print('Remove old file...')
    currentProject.deleteJsonEmailData(user)
    print('Create new file')
    currentProject.saveJsonEmailData(user, json_email_data)

def update_predictions_db(name_model='mnb'):
    currentProject = MLProject()  # --- this part could be improved
    threshold_dic = currentProject.get_threshold_dic()
    estimator = currentProject.loadBestEstimator(name_model)

    print('Update predictions for model :'+str(name_model))
    update_dic = get_new_predictions(estimator,threshold=threshold_dic[name_model])
    update_predictions(update_dic)
    print('Predictions updated!')