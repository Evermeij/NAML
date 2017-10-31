from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import Flask, send_file, request, jsonify, render_template,redirect
from werkzeug.utils import secure_filename

import time
import sys,os

#sys.path.append('app/')
#sys.path.append('database/')
#sys.path.append('machine_learning/')
#sys.path.append('test/')

from database.postgresDB import init_database,load_user_mails,update_user_mails,delete_user_mails,update_mail_database,correct_predictions_from_input
from machineLearning.config import load_filenames_images, get_Email_names
from machineLearning.generation import generate_new_images_manual_fit,generate_new_images_auto_fit,add_new_email_images,update_predictions_db
from machineLearning.config import censor_name,\
                                   load_censored_words,\
                                   reset__censored_words,\
                                   update_censored_words

#UPLOAD_FOLDER = '/emails/msg/'
UPLOAD_FOLDER_EML = '/emails/eml/'
UPLOAD_FOLDER_MSG = '/emails/msg/'
ALLOWED_EXTENSIONS = set(['msg','eml'])

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_EML'] = UPLOAD_FOLDER_EML
app.config['UPLOAD_FOLDER_MSG'] = UPLOAD_FOLDER_MSG

app.config['SECRET_KEY'] = 'bigdatarepublic@secretkey2017'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////database.db'
Bootstrap(app)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

current_model_name = None

localdir = '/app'
########################################################################################################################
#LOG IN CLASSES
########################################################################################################################
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15))#, unique=True)
    email = db.Column(db.String(50))#, unique=True)
    password = db.Column(db.String(80))

#######################################
# ADD ADMIN UPON START
def database_initialization_sequence():
    print('Adding User')
    db.create_all()
    hashed_password = generate_password_hash('bdr@admin', method='sha256')
    admin = User(username = 'admin',email = 'benoit.descamps@bigdatarepublic.nl',password = hashed_password)
    print(admin.username)
    print(admin.email)
    print(admin.password)
    db.session.add( admin )

    print('Adding User')
    db.create_all()
    hashed_password = generate_password_hash('na@mette', method='sha256')
    admin = User(username='mette', email='Mette.van.Essen@nationaalarchief.nl', password=hashed_password)
    print(admin.username)
    print(admin.email)
    print(admin.password)
    db.session.add(admin)

    #db.session.rollback()
    db.session.commit()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])

########################################################################################################################
#UPLOAD FUNCTIONS
########################################################################################################################

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

########################################################################################################################
#EXTRA FUNCTIONS
########################################################################################################################
def create_user_email_im_dir(username):
    print('Creating directory of email images of '+str(username) )
    dir_path = localdir+'/static/Images/Emails/Users/'
    directory = username
    if not os.path.exists(dir_path+directory):
        os.makedirs(dir_path+directory+'/'+'feature_importance_email_NA')
        os.makedirs(dir_path+directory+'/'+'feature_importance_email_NA'+'/'+ 'etr')
        os.makedirs(dir_path+directory+'/'+'feature_importance_email_NA'+'/'+ 'mnb')
        os.makedirs(dir_path + directory + '/' + 'feature_importance_email_NA' + '/' + 'rf')
        os.makedirs(dir_path+directory+'/'+'pie_probability_NA')
        os.makedirs(dir_path+directory+'/'+'pie_probability_NA'+'/'+ 'etr')
        os.makedirs(dir_path+directory+'/'+'pie_probability_NA'+'/'+ 'mnb')
        os.makedirs(dir_path + directory + '/' + 'pie_probability_NA' + '/' + 'rf')
        with open(dir_path+directory+'/json_email_data_NA.txt',"w") as email_data_file:
            email_data_file.write("{}")

########################################################################################################################
#REQUESTS HANDLING
########################################################################################################################
@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            print('I am here...')
            print(check_password_hash(user.password, form.password.data) )
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect('/controleboard/')
        return '<h1>Invalid username or password</h1>'


    return render_template('login.html', form=form)

@app.route('/controleboard/',methods=['GET','POST'] )
@login_required
def index():
    print('Current User: ')
    print(current_user.username)

    print(request.method)
    if request.method == 'POST':
        print('Caught post request...')
        post_data = request.get_json()
        print(post_data)

        correct_predictions_from_input(post_data['mail_id'],post_data['truth_class'] )
        # with open(localdir+'/static/data/corrections/corr.txt','a') as file:
        #     file.write( str(post_data['mail_id']+' ; ' +str(post_data['truth_class'])) )
        #     file.write('\n')
    if current_user.username == 'admin':
        return send_file("templates//index_controleboard_admin.html")
    else:
        return send_file("templates/index_controleboard.html")



@app.route('/signup/', methods=['GET', 'POST'])
@login_required
def signup():
    if current_user.username != 'admin':
        return redirect('/')
    form = RegisterForm()

    if form.validate_on_submit():
        print('Creating new User...')
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        create_user_email_im_dir(new_user.username)

        return '<h1>New user has been created!</h1>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

#add temporarily
@app.route('/corrections/', methods=['GET'])
@login_required
def get_corrections():
    """
    View which is called whenever the '/s/' this url is requested
    """
    try:
        file_corr = localdir +'/static/data/corrections/corr.txt'
        with open(file_corr, 'r') as file:
            corrections = file.read()
    except:
        print('file not found...')
        corrections = ''
        pass
    return jsonify(corrections)

@app.route('/s/', methods=['GET'])
@login_required
def search_query():
    """
    View which is called whenever the '/s/' this url is requested
    """
    return jsonify(load_user_mails(username=current_user.username,address=current_user.email))

@app.route('/global_performances/',methods=['GET','POST'] )
@login_required
def index_performance():
    images = dict()
    images['pie'] = ''
    if request.method == 'POST':
        post_data = request.get_json()

        if 'new_model' in post_data.keys():####---------> is dit stuk nog nodig?????
            current_model_name = post_data['new_model']
            print('Update model name '+ current_model_name + ' ...')
            post_filenames()

        if 'message' in post_data.keys():
            print(post_data)
            if(post_data['message']== 'TRAIN'):
                print(post_data)
                generate_new_images_manual_fit(name_model =post_data['model_name'],
                                            thres = float(post_data['threshold']),
                                            weight_taak = float(post_data['weight_taak']),
                                            weight_non_taak = float(post_data['weight_non_taak']) )
            if (post_data['message'] == 'AUTO_TRAIN'):
                print('Start Grid Search...')
                generate_new_images_auto_fit(name_model=post_data['model_name'])
    if current_user.username == 'admin':
        return send_file("templates/index_global_performance_admin.html")
    else:
        return send_file("templates/index_global_performance_user.html")



#######################################################################################################################
@app.route('/images/',methods=['GET','POST'] )
@login_required
def post_filenames():

    filenames_dict = load_filenames_images()
    return jsonify(filenames_dict)
########################################################################################################################

@app.route('/email_data/',methods=['GET','POST'] )
@login_required
def post_emailnames():
    if request.method == 'POST':
        post_data = request.get_json()
        if 'mail_id' in post_data.keys():
            new_mail_id = post_data['mail_id']
            print('Update email data and images of '+str(current_user.username))
            #print( ml.get_mail_test(new_mail_id) )
            add_new_email_images(new_mail_id, user=current_user.username)
        if 'message' in post_data.keys():
            if post_data['message'] == 'RESET':
                def clean_dir(pathdir, extra_dir='/'):
                    '''
                    :param pathdir: 
                    :return: deletes all .png and .txt within the dir
                    '''
                    for filename in os.listdir(pathdir + extra_dir):
                        if (filename.split('.')[1] == 'txt') or (filename.split('.')[1] == 'png'):
                            print('Deleting File: ' + str(filename))
                            os.remove(pathdir + extra_dir + filename)

                dir_path = localdir +'/static/Images/Emails/Users/'
                directory = current_user.username
                clean_dir(dir_path + directory + '/' + 'feature_importance_email_NA' + '/' + 'etr/')
                clean_dir(dir_path + directory + '/' + 'feature_importance_email_NA' + '/' + 'mnb/')
                clean_dir(dir_path + directory + '/' + 'feature_importance_email_NA' + '/' + 'rf')
                clean_dir(dir_path + directory + '/' + 'pie_probability_NA' + '/' + 'etr/')
                clean_dir(dir_path + directory + '/' + 'pie_probability_NA' + '/' + 'mnb/')
                clean_dir(dir_path + directory + '/' + 'pie_probability_NA' + '/' + 'rf/')

                os.remove(dir_path + directory + '/' + 'json_email_data_NA.txt')
                with open(dir_path + directory + '/' + 'json_email_data_NA.txt', "w") as email_data_file:
                    email_data_file.write("{}")

    return jsonify(get_Email_names(current_user.username))


@app.route('/emails/',methods=['GET','POST'] )
@login_required
def index_emails():
    return send_file("templates/index_email.html" )

@app.route('/logout')
@login_required
def logout():
    delete_user_mails(current_user.username)
    logout_user()
    return redirect('/')

@app.route('/upload_emails/',methods=['GET','POST'] )
@login_required
def index_upload_emails():
    if current_user.username != 'admin':
        return redirect('/')
    if request.method == 'POST':
        print(request.files)
        if 'file' in request.files:
            uploaded_files = request.files.getlist("file")
            for file in uploaded_files:
                if file and allowed_file(file.filename):
                    extension = file.filename.split('.')[-1]
                    print('filename: '+str(file.filename))
                    print('extension: '+str(extension))
                    filename = secure_filename(file.filename)
                    if (extension == 'eml'):
                        file.save(os.path.join(app.config['UPLOAD_FOLDER_EML'], filename))
                    if (extension == 'msg'):
                        file.save(os.path.join(app.config['UPLOAD_FOLDER_MSG'], filename))
    return redirect('/global_performances/')

@app.route('/update_database/',methods=['GET','POST'] )
@login_required
def index_update_database():
    if request.method == 'POST':
        post_data = request.get_json()
        if 'message' in post_data.keys():
            if post_data['message'] == 'UPDATE_DATABASE':
                update_mail_database( path_mails='/emails/processed/')
                for filename in os.listdir('/emails/processed/'):
                    if (filename.split('.')[1] == 'txt') or (filename.split('.')[1] == 'png'):
                        print('Deleting File: ' + str(filename))
                        os.remove('/emails/processed/'+ filename)
    return redirect('/')

@app.route('/update_predictions/',methods=['GET','POST'] )
@login_required
def index_update_predictions():
    if request.method == 'POST':
        post_data = request.get_json()
        if 'message' in post_data.keys():
            if (post_data['message'] == 'UPDATE_PREDICTIONS'):
                print('UPDATE_PREDICTIONS')
                name_model = post_data['model_name']
                update_predictions_db(name_model)

    return redirect('/')

@app.route('/censored/',methods=['GET','POST'] )
@login_required
def index_update_censored():
    if request.method == 'POST':
        post_data = request.get_json()
        if 'message' in post_data.keys():
            if (post_data['message'] == 'UPDATE_CENSORED_WORDS'):
                print('UPDATE_CENSORED_WORDS')
                new_censored_words = post_data['words']
                list_new_censored = list( map( lambda x: x.replace(' ',''), new_censored_words.split(';')))
                update_censored_words(list_new_censored)
            if (post_data['message'] == 'RESET_CENSORED_WORDS'):
                print('RESET_CENSORED_WORDS')
                reset__censored_words()

    return redirect('/')

@app.route('/refresh/',methods=['GET','POST'] )
@login_required
def index_refresh_list():
    if request.method == 'POST':
        post_data = request.get_json()
        if 'message' in post_data.keys():
            if (post_data['message'] == 'REFRESH_LIST'):
                update_user_mails(username=current_user.username,address=current_user.email)

    return redirect('/')


#temporary solution for gunicorn compatibility
# better solution in http://flask.pocoo.org/docs/0.12/tutorial/dbcon/
@app.before_first_request
def setup_logging():
    print('Create new table...')
    init_database()

    print('Done...')
    dbstatus = False
    while dbstatus == False:
        try:
            db.create_all()
        except:
            time.sleep(2)
        else:
            dbstatus = True
    database_initialization_sequence()

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')#,port=80) #add port for discipl
    #app.run(host='0.0.0.0',port=80)