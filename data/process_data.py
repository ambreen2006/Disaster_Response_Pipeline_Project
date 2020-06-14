import sys
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Loads messages and categories data from csv files'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    '''Clean and organize data'''
    def get_categories_as_dataframe(data):
        cats = data.categories.str.split(';', expand = True)
        row = cats.iloc[0,:]
        category_colnames = row.apply(lambda x: x[:-2])
        cats.columns = category_colnames
        return cats

    def clean_categories(data):
        cats = get_categories_as_dataframe(data)
        for column in cats:
            cats[column]  = cats[column].str[-1]
            cats[column] = cats[column].astype('int')
        cats.replace(to_replace = 2, value = 1, inplace = True)
        categories.drop(columns = ['child_alone'], inplace = True)
        data = pd.concat([data.drop(columns = ['categories']), cats],
                        axis = 1)
        return data

    def remove_duplicates(data):
        '''Removes duplicates from the data'''
        data = data.drop_duplicates()
        return data

    return remove_duplicates(clean_categories(df))


def save_data(df, database_filename):
    '''
    Saves the dataframe to the filename provided. The filename MUST NOT
    include the sql:/// prefix
    '''
    uri = 'sqlite:///' + database_filename
    engine = create_engine(uri)
    df.to_sql('Messages', engine, index = False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
