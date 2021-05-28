import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

# import statements
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import pickle

# a regular expression for urls
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """This function loads data from a database and returns formated data for ML applications.

    Input:
    database_filepath -- the filepath to the database (without sqlite prefix), e.g. /data/RobsDisasterResponse.db
    
    Output:
    X -- the X data for ML applications. Here the messages (text body) from the dataset
    y -- the y data for ML applications. Here the multilabel classification for the X data
    data_top -- the names of the categories
    """
    # load data from database
    database_filepath = 'sqlite://' + database_filepath
    engine = create_engine(database_filepath)
    
    # crate dataframe
    df = pd.read_sql_table('RobsMessages', engine)
    
    # dropping columns with all 0 entries as those contain no information for the model
    df = df.loc[:, (df != 0).any(axis=0)]

    # create output data
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    
    # make sure label data is integer
    y = y.astype('int')
    
    # get category names
    df_categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    data_top = list(df_categories.columns.values)
    
    return X, y, data_top

def tokenize(text):
    """This function normalizes, lemmartizes and tokenizes text for ML applications.

    Input:
    text -- some text to be transformed, e.g. 'This is text'
    
    Output:
    clean_tokens -- clean tokens from the text
    """
    
    # remove all urls and replace it with placeholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # word tokenize
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    # strip, to lower case and lemmatize
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """This function builds a ML model using a pipeline.

    Input:
    none
    
    Output:
    cv -- the model using a pipeline and using grid search  
    """
    pipeline = Pipeline([
        #('vect', CountVectorizer(strip_accents='ascii', tokenizer=tokenize, stop_words='english', max_df=0.7, binary=True)),
        #('vect', CountVectorizer(strip_accents='ascii', tokenizer=tokenize, max_df=0.7, binary=True)),
        ('vect', CountVectorizer(strip_accents='ascii', tokenizer=tokenize, binary=True)),
        ('tfidf', TfidfTransformer()),
        #('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1), n_jobs=-1))
        #('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None), n_jobs=-1))
        #('clf', OneVsRestClassifier(SVC(kernel='linear')))
        ('clf', OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5)))
    ])
    
    parameters = {
    #    'vect__ngram_range': ((1, 1), (1, 2)),
    #    'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
    #    'tfidf__use_idf': (True, False),
        'clf__estimator__dual': [True, False],
    #    'clf__estimator__max_iter': [500, 1000]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function evaluetes a ML model with the classification_report function.

    Input:
    model -- the model to be evaluated
    X_Test -- the X data for the test
    Y_test -- the truth data
    category_names -- the names of the categories for multi-label classification
    
    Output:
    none, only printout of report
    """
    
    # predict on test data
    y_pred = model.predict(X_test)
    
    # print classification report
    print(classification_report(Y_test, y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    """This function saves the model to the specified file path.

    Input:
    model -- the model to be saved
    model_filepath -- the file path where the model is to be saved
    
    Output:
    none, only saves model to destination
    """
    
    # save the model to disk
    filename = 'robs_finalized_model.pkl'
    model_filepath = model_filepath + '/' + filename
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """The main function. Tha data is loaded, data is split into train and test, model is built,
    fitted and evaluated. The best parameters are printed out and the model is saved to disk.

    Input:
    none, database_filepath and model_filepath have to be specified
    
    Output:
    none, only saves model to destination
    """
    if len(sys.argv) == 3:
        
        # get the parameters
        database_filepath, model_filepath = sys.argv[1:]
        
        # load the data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        # split the data in train and test set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # build the model
        print('Building model...')
        model = build_model()
        
        # fit the model
        print('Training model... Get a coffee this will take a while')
        model.fit(X_train, Y_train)
        
        # evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
#        print('Improving model...')
#        cv = improve_model(model)
        
#        print('Refitting model... Get a coffee this will take a while (forever)')
#        cv.fit(X_train, Y_train)
        
        # print the best parameters found
        print('Best parameters found:')
        print(model.best_params_)
        
#        print('Evaluating improved model...')
#        evaluate_model(cv, X_test, Y_test, category_names)

        # save the model
        print('Saving model...\n    Model Path: {}'.format(model_filepath))
        # save_model(model, model_filepath)
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()