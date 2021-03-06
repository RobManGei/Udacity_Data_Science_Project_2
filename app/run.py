import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """This function normalizes, lemmartizes and tokenizes text for ML applications.

    Input:
    text -- some text to be transformed, e.g. 'This is text'
    
    Output:
    clean_tokens -- clean tokens from the text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/RobsDisasterResponse.db')
df = pd.read_sql_table('RobsMessages', engine)

# dropping columns with all 0 entries as those contain no information for the model
# this has to be done here, as it is done in the ML pipeline. Otherwise the categories do not match
df = df.loc[:, (df != 0).any(axis=0)]

# load model
model = joblib.load("../models/robs_finalized_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
        
    # prepare data for genre count plot
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    print(genre_names)
    genre_counts = genre_counts.iloc[:].values
    print(genre_counts)
    
    # prepare data for number of labels taged plot
    rowsums = df.iloc[:,3:].sum(axis=1)
    y_rowsums=rowsums.value_counts().sort_index()
    x_rowsums=list(range(0, len(y_rowsums)))
    
    # prepare data for number of occurances for a label
    colsums = df.iloc[:,4:].sum(axis=0)
    y_colsums = colsums.iloc[:].values
    x_colsums = colsums.index
         
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=x_rowsums,
                    y=y_rowsums
                )
            ],

            'layout': {
                'title': 'Number of labels identified per message',
                'yaxis': {
                    'title': "Number of occurrences"
                },
                'xaxis': {
                    'title': "Number of labels for a message"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_colsums,
                    y=y_colsums
                )
            ],

            'layout': {
                'title': 'Number of occurrences per label',
                'yaxis': {
                    'title': "Number of occurrences"
                },
                'xaxis': {
                    'title': "Label name"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
    

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """The main function.Starts webserver.

    Input:
    none
    
    Output:
    none
    """
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()