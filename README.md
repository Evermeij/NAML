# National Archive Machine Learning Experiment

Demo-Prototype code.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* [Docker](https://www.docker.com/)

### Installing


```
docker-compose build
```

### Run
```
docker-compose up
```
## Authors

* **Benoit Descamps** - *Initial work* - [github](https://github.com/benoitdescamps), benoitdescamps@hotmail.com

## Background
The goal of this project is the development of a demo-prototype to illustrate the inner-workings of a basic text-classification,i.e. binary classification (TAAK vs NON-TAAK) of Email Bodies,  machine learning product.

When running the app, the User first encounters a login.

<p align="center">
  <img src="/img/screenshotlogin.PNG"/>
</p>

At start, two users are created:
* admin, password: bdr@admin
* mette, password: na@mette

The admin has the possibily to navigate to /signup and create new users.
<p align="center">
  <img src="/img/screenshotSignup.PNG"/>
</p>

After identification, the User faces a "Controle Board":
<p align="center">
  <img src="/img/screenshot_main.PNG"/>
</p>

On this board, the User has access to all uploaded emails (see login as admin and navigate to the Model-Board ) if his email is present in From or To.
The admin has access to all emails and has the possibily to censor names/words if necessary.

The User sees how each email is classified by the currently running machine learning model. He has the possibility to confirm the prediction or correcting it. After submission the current flag of the email will be updated, and this correction will be taken into account by the machine learning algorithm.

The User can also press the  "Detail"-button and navigate to the "Email-board":

<p align="center">
  <img src="/img/screenshot_email.PNG"/>
</p>

In the text, words are highlighted in function of its "feature importance" according to the used algorithm.
* For [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html): This is the Empirical log probability of features given a class.
* For [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): This is frequency of the features (word or n-gram) at the split.
* For [Extreme Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html): Idem

In the Model-Board, the admin has been given a certain controle over the recalibration of the models:

<p align="center">
  <img src="/img/screenhost_admin.PNG"/>
</p>
The admin has the possibility to upload additional emails, update the database and update the target-predictions according the currently chosen model.
Finally the admin has the possibility to explore the current performance of the models by retrainen the model manually or automatically.
The automatic recalibration is a random grid-search with 10-iterations over 3 cross-validation folds over values of three types of hyperparameters: 
* Weight Taak/Non-Taak: Due to presence of imbalance, the sample weight of each class is controlled with these hyperparameters
* Threshold: Shift the class choice with class_0 = (Score < threshold), class_1 = (Score >= threshold).


## License

This project is licensed under ?????

## Acknowledgments

* Dutch National Archive for initiating the project
* [BigData Republic](https://www.bigdatarepublic.nl/)
* ICTU
* Discipl

