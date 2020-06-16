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

You can see that we have unbalanced data, as related is more than 75% followed
by spikes in few more Categories such as aid_related and weather_related.

![Data Distribution](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/data_distribution.png)

There are few ways to handle unbalanced data and I have found undersampling to
work before however for the scope of this project, I've attempted to select a model
providing better recall.

## ML (ML Pipeline Preparation.ipynb, models/train_classifier.py)

Different classifiers are compared against `AUC`, `f1-score`, and `recall` score.
Because the data is unbalanced, I set scale_pos_weight for XGBoost.

XGBoost appears to outperform the other
classifiers but with a setting of `scale_pos_weight` set to 10. Just with the default
setting, I see AdaBoost outperforming otherwise.

![Models Comparison](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/multiple_models_result.png)

XGBoost classifier is then further optimized using GridSearchCV for better recall.

![Results after GridSearchCV](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/optimized_model_result.png)

However when visually inspecting the prediction based on non-seen data, the optimized model seems too eager to classify and
choose a lot more tags then seemingly necessary.

For example predictions on the following text from Britannica are depicted below:

```
The floods, which affected approximately 20 million people, destroyed homes, crops, and infrastructure and left millions vulnerable to malnutrition and waterborne disease.
```

| XGBoost | GridSearchCV - XGBoost |
|---|---|
|![Model Prediction](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/model_britannica_prediction.png)|![Optimized Model Prediction](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/model_optimized_britannica_prediction.png)

Predictions on the following text from NYTimes tech section are depicted below:

| XGBoost | GridSearchCV - XGBoost |
|---|---|
|![Model Prediction](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/model_nytimes_tech_column_prediction.png)| ![Optimized Model Prediction](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/model_optimized_nytimes_tech_column_predictions.png)

Neither of the two are perfect but the conservative predictions I believe would
be more useful to the aid worker.

Therefore for the flask app, I went with the XGBoost model with `scale_pos_weight` as `10` and hence the train_classifier.py uses the XGBoost with `scale_pos_weight` set to 10.

Run the script as follows:

`python train_classifier.py <db-path> <path-to-persist-classifier>`

I would like to note that running the XGBoost model again resulted in slightly
different scores perhaps a remedy would be to use seed value.

![XGBoost score](https://github.com/ambreen2006/Disaster_Response_Pipeline_Project/blob/master/Resources/simple_model_results.png)

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
