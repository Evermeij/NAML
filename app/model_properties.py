from wtforms import Form,FloatField, validators, IntegerField
import os

path_confusion_matrix = os.getcwd() + '/../app/static/Images/confusion_matrices_NA/'
path_wordcloud = os.getcwd() + '/../app/static/Images/wordcloud_NA/'
path_pies = os.getcwd() + '/../app/static/Images/pies_NA/'
path_rocs = os.getcwd() + '/../app/static/Images/rocs_NA/'
path_models = os.getcwd()+'/../app/static/Images/models_NA/'

path_im_dic = {'cm':"/../static/Images/confusion_matrices_NA/",
            'wc': "/../static/Images/wordcloud_NA/",
            'pie': "/static/Images/pies_NA/",
            'roc': "/../static/Images/rocs_NA/" }

class MNBInputForm(Form):
    WeightTaak = FloatField(
        label='gewicht Taak', default=0.49,
        validators=[validators.InputRequired()])
    WeightNonTaak = FloatField(
        label='gewicht Non Taak', default=0.50,
        validators=[validators.InputRequired()])
    Thres = FloatField(
        label='threshold', default=0.44,
        validators=[validators.InputRequired()])

class RFInputForm(Form):
    Trees = IntegerField(
        label='trees', default=50,
        validators=[validators.InputRequired()])
    WeightTaak = FloatField(
        label='gewicht Taak', default=0.49,
        validators=[validators.InputRequired()])
    WeightNonTaak = FloatField(
        label='gewicht Non Taak', default=0.50,
        validators=[validators.InputRequired()])
    Thres = FloatField(
        label='threshold', default=0.49,
        validators=[validators.InputRequired()])

class ETRInputForm(Form):
    Trees = IntegerField(
        label='trees', default=50,
        validators=[validators.InputRequired()])
    WeightTaak = FloatField(
        label='gewicht Taak', default=0.49,
        validators=[validators.InputRequired()])
    WeightNonTaak = FloatField(
        label='gewicht Non Taak', default=0.50,
        validators=[validators.InputRequired()])
    Thres = FloatField(
        label='threshold', default=0.392,
        validators=[validators.InputRequired()])

########################################################################################################################
def get_file(path,type, extension = 'png'):
    filename = ''
    print(path)
    for filename in os.listdir(path):
        print(filename)
        if (filename.split('.')[1]== 'png'):
            print('Get File : '+str(filename) )

    return path_im_dic[type] + filename
def get_current_path(path,model_name):
    return path + model_name + '/'
def update_model_performance_image(name):
    images = dict()
    if name == 'mnb':
        images['cm'] = get_file( path = get_current_path(path_confusion_matrix,'mnb'), type = 'cm')
        images['wc'] = get_file( path = get_current_path(path_wordcloud, 'mnb') ,type = 'wc')
        images['pie'] =  get_file( path = get_current_path(path_pies, 'mnb'), type = 'pie' )
        images['roc'] =  get_file( path = get_current_path(path_rocs, 'mnb'),type = 'roc'  )

    if name == 'rf':
        images['cm'] = get_file( path = get_current_path(path_confusion_matrix,'rf'),type = 'cm' )
        images['wc'] = get_file( path = get_current_path(path_wordcloud, 'rf'),type = 'wc' )
        images['pie'] =  get_file( path = get_current_path(path_pies, 'rf'),type = 'pie'  )
        images['roc'] =  get_file( path = get_current_path(path_rocs, 'rf'),type = 'roc' )
    if name == 'etr':
        images['cm'] = get_file( path = get_current_path(path_confusion_matrix,'etr'),type = 'cm' )
        images['wc'] = get_file( path = get_current_path(path_wordcloud, 'etr'),type = 'wc' )
        images['pie'] =  get_file( path = get_current_path(path_pies, 'etr'),type = 'pie'  )
        images['roc'] =  get_file( path = get_current_path(path_rocs, 'etr'),type = 'roc' )

    return images
