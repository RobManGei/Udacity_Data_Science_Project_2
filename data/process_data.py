# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """This function loads data from two csv files and returns a pandas dataframe.

    Input:
    messages_filepath -- the filepath to the csv file with the messages
    categories_filepath -- the filepath to the csv file with the categories
    
    Output:
    df -- a pandas dataframe with the merged data
    """
    # read messages
    messages = pd.read_csv(messages_filepath)
    
    # read categories
    categories = pd.read_csv(categories_filepath)
    
    # merge data in a pandas dataframe
    df = messages.merge(categories, left_on='id', right_on='id', how='outer')
    
    return df


def clean_data(df):
    """This function cleans the data in a pandas dataframe. Categories are transformed into column names
    and the category labels are reduced to 0s and 1s and tuned into numeric variables

    Input:
    df -- a pandas dataframe
    
    Output:
    df -- the cleaned pandas dataframe
    """
    
    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split("-").str.get(1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #there were some 2s in there
    categories = categories.replace(2,1)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset="id", keep = False, inplace = True)

    return df

def save_data(df, database_filename):
    """This function saves the dataframe to a sql database.

    Input:
    df -- the pandas dataframe to be saved
    database_filename -- the file path and name where the database is to be saved
    
    Output:
    none, only saves database to destination
    """
    
    # create vald filename
    database_filename = 'sqlite://' + database_filename
    
    # crate an engine
    engine = create_engine(database_filename)
    
    # store database
    df.to_sql('RobsMessages', engine, index=False, if_exists='replace')


def main():
    """The main function. Tha data is loaded, cleaned and saved to disk.

    Input:
    none, messages_filepath, categories_filepath and database_filepath have to be specified
    
    Output:
    none, only saves database to destination
    """
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