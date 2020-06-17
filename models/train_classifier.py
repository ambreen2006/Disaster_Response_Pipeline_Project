# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sqlalchemy import create_engine

import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loads data from the sql database

    Args:
    -----
    database_filepath: the relative file path of the database. The name MUST
    NOT include sql:// as it is appended in this method

    Returns:
    --------
    X: The message feature data
    Y: The target labels data
    column names: Names of the target labels
    """

    path = 'sqlite:///'+database_filepath
    engine = create_engine(path)
    df = pd.read_sql('select * from Messages', engine)
    df.drop(columns = ['id', 'original'], inplace = True)
    X = df[['message', 'genre']]
    Y = df.loc[:, ~df.columns.isin(X.columns)]
    return X['message'], Y, Y.columns.values

def tokenize(text):
    """Replace url with a placeholder then tokenize and lemmatize words in the text"""

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(x).strip() for x in tokens
                                            if x not in stop_words]
    return tokens

def scorer_f1(yt, yp):
    """
    Returns f1_score with desired parameters.
    For testing purposes, if a smaller sample of records is passed,
    then the prediction may not have all the classifications resulting in
    zero division warning.

    Args:
    -----
    yt: the true target
    yp: the predicted categories

    Returns:
    --------
    f1 score
    """
    
    return f1_score(yt, yp, average='macro', zero_division=1)

def scorer_recall(yt, yp):
    """
    Returns recall_score with desired parameters.
    For testing purposes, if a smaller sample of records is passed,
    then the prediction may not have all the classifications resulting in
    zero division warning.

    Args:
    -----
    yt: the true target
    yp: the predicted categories

    Returns:
    --------
    recall score
    """    
    
    return recall_score(yt, yp, average='macro', zero_division=1)

def build_model():
    """Search for better XGBoost parameters using GridSearchCV"""
    
    def get_best_classifier(selected_pipeline, parameters):

        selected_scorers = {
            'f1': make_scorer(scorer_f1),
            'recall': make_scorer(scorer_recall)
        }

        # I'm not using n_jobs=-1 because it seems to be that
        # there is some issue in the library regarding leaks
        cv = GridSearchCV(selected_pipeline, param_grid=parameters,
                            verbose=1,
                            scoring=selected_scorers,
                            refit='recall')
        return cv

    selected_pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultiOutputClassifier(XGBClassifier()))
    ])

    parameters = {
        'classifier__estimator__scale_pos_weight':[5, 10, 20, 100],
        'classifier__estimator__learning_rate': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
        'tfidf__use_idf': [True, False],
    }

    cv = get_best_classifier(selected_pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model on recall, auc, and f1-score and prints it on the screen.
    
    Args:
    -----
    model: the fitted scikit model to be evaluated
    X_test: the input features on which predictions are to be made
    Y_test: the actual target categories
    category_names: the category names

    """

    def labeled_classification_report(Y_test, y_pred):
        '''display output of classification_report for each label'''
        for i in range(0, len(category_names)):
            print(f'{category_names[i]}')
            print('-'*55)
            c_pred = y_pred[:,i]
            c_true = Y_test[category_names[i]]
            print(classification_report(c_true, c_pred, zero_division=1))

    def recall_f1_auc(y_true, predicted):
        '''Calculates recall, f1, and auc scores'''

        f1 = f1_score(y_true, predicted, average = 'macro', zero_division=1)
        auc = roc_auc_score(y_true, predicted, multi_class = 'ovo', average ='macro')
        recall = recall_score(y_true, predicted, average = 'macro', zero_division=1)
        return f1, auc, recall

    def display_scores(y_true, predicted):
        '''Display f-1, recall, and AUC scores'''

        f1, auc, recall = recall_f1_auc(y_true, predicted)
        print('======================')
        print(f'recall score: {recall}')
        print(f'f1 score: {f1}')
        print(f'AUC score: {auc}')
        print('======================')

    y_pred = model.predict(X_test)
    print("=== Estimator Information")
    print(model.best_estimator_)
    print(model.best_params_)

    print("=== Evaluation")
    display_scores(Y_test.values, y_pred)
    labeled_classification_report(Y_test, y_pred)
    return

def save_model(model, model_filepath):
    """
    Pickle the model to the path specified.

    Args:
    -----
    model: model to be saved
    model_filpath: path indicating where to save the model
    """

    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
