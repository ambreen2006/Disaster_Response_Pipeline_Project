import re
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
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

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''
    Loads data from the sql database

    Args:
    -----
    database_filepath: the relative file path of the database. The name MUST
    NOT include sql:// as it is appended in this method

    Returns:
    X: The message feature data
    Y: The target labels data
    column names: Names of the target labels
    '''

    path = 'sqlite:///'+database_filepath
    engine = create_engine(path)
    df = pd.read_sql('select * from Messages', engine)
    df.drop(columns = ['id', 'original'], inplace = True)
    X = df[['message', 'genre']]
    Y = df.loc[:, ~df.columns.isin(X.columns)]
    return X['message'], Y, Y.columns.values

def tokenize(text):
    '''
    Replace url with a placeholder then tokenize and lemmatize words in the
    text
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(x).lower().strip() for x in tokens]
    return tokens


def build_model():
    '''
    Get a tuned pipeline using GridSearchCV. AdaBoostClassifier is choosen
    becaused based on the analysis done on `ML Pipeline Preparation.ipynb`, it
    performed better in terms of f1-score and AUC.
    '''

    def get_best_classifier(selected_pipeline, parameters):
          cv = GridSearchCV(selected_pipeline,
                            param_grid = parameters,
                            verbose = 1,
                            n_jobs = -1)
          return cv

    selected_pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'classifier__estimator__learning_rate': [0.2, 0.4, 0.6, 0.8, 1],
        'tfidf__use_idf': [True, False],
        'classifier__estimator__algorithm':['SAMME', 'SAMME.R'],
        'classifier__estimator__n_estimators': [5, 10, 15, 20, 30, 40, 50],
    }

    cv = get_best_classifier(selected_pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    def labeled_classification_report(Y_test, y_pred):
        '''display output of classification_report for each label'''

        for i in range(0, len(category_names)):
            print(f'{category_names[i]}')
            print('-'*55)
            c_pred = y_pred[:,i]
            c_true = Y_test[category_names[i]]
            print(classification_report(c_true, c_pred))

    def display_scores(y_true, predicted):
        '''Display f-1 and AUC scores'''

        f1 = f1_score(y_test, predicted, average = 'weighted')
        auc = roc_auc_score(y_true, predicted, multi_class = 'ovo',
                            average ='weighted')
        display(f'f1-score: {f1}')
        display(f'AUC score: {auc}')

    y_pred = model.predict(X_test)
    print("=== Estimator Information")
    print(model.best_estimator_)
    print(model.best_params_)

    print("=== Evaluation")
    display_scores(Y_test.values, y_pred)
    labeled_classification_report(y_test, y_pred1)
    return


def save_model(model, model_filepath):
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
