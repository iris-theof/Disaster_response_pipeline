import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data//DisasterResponse.db')
df = pd.read_sql_table('disaster_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_per = round(100*genre_counts/genre_counts.sum(), 2)
    genre_names = list(genre_counts.index)
    
    category_related_counts = df.groupby('related').count()['message']
    category_related_names = ['Related' if i==1 else 'Not Related' for i in list(category_related_counts.index)]

    requests_counts = df.groupby(['related','request']).count().loc[1,'message']
    category_requests_names = ['Requests' if i==1 else 'Not Requests' for i in list(requests_counts.index)]
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values

    # create visuals
    graphs = [
           {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_per,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "#7fc97f",
                    "#beaed4",
                    "#fdc086"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_counts
              }
            ],
            "layout": {
              "title": "Distribution of Messages by Genre"
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "",
                    'tickangle': 35
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_related_names,
                    y=category_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related with Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_requests_names,
                    y=requests_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Request Messages <br> out of all Disaster Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
