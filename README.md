# Disaster Response Pipeline Project

## Objective
This is a Udacity project, classifiying messages related to natural disaster into different tags. This can help organizers by
saving time in categorizing messages and expediate the appropriate response.

## Setup

Dependencies includes:

* Flask
* Pandas
* scikit
* XGBoost

## ETL (ETL Pipeline Preparation.ipynb, data/process_data.py)

Categories are cleaned and converted into mult-label binary representation. Categories which has no message belonging to it are 
dropped.

Run as follows:

`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

The clean data frame is then written to a sql database.

## ML (ML Pipeline Preparation.ipynb, models/train_classifier.py)

Different classifiers are compared against `AUC`, 'f1-score`, and `recall` score. XGBoost appears to outperform the other 
classifiers but with a setting of `scale_pos_weight` set to 10.

XGBoost classifier is then further optimized for better recall.

However when visually inspecting the prediction based on non-seen data, the optimized model seems too eager to classify and 
choose a lot more tags then seemingly necessary, therefore for the flask app, I went with the XGBoost model with `scale_pos_weight` as `10`

Therefore the train_classifier.py uses the XGBoost with `scale_pos_weight` set to 10.

Run as follows:

`python train_classifier.py <db-path> <path-to-persist-classifier>`

## Web App

Run as follows:

```
python run.py
# in another terminal window
env|grep WORK
```
In a new web browser window, type in the following while replacing SPACEID and SPACEDOMAIN from the output of the previous 
command:

`https://SPACEID-3001.SPACEDOMAIN`


