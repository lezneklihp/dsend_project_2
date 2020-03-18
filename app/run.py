from flask import Flask
from flask import render_template, request, jsonify
import json
from lightgbm import LGBMClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import plotly
from plotly.graph_objs import Bar, Pie
import re
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)


def tokenize(corpus):
    """Tokenizes an English corpus (a text).

    :param df: pandas.core.frame.DataFrame

    :return: pandas.core.series.Series

    >>> tokenized_words = tokenize(corpus)
    """

    # Normalize corpus
    text = corpus.lower()

    # Tokenize corpus; This will give us a list of lists of words
    words_all = word_tokenize(text)

    # Remove stop-words in the list of lists of words
    words_all = [word for word in words_all if word not in stopwords.words('english')]

    # Replace left-overs from colloquial abbreviations, such as in "you've"
    left_overs = r'|'.join((r"'m", r"'s", r"n't", r"'d", r"'re", r"'ve", r"'t", r"'ll", r"'nt", r"'nt"))
    words_all = re.sub(left_overs, "", str(words_all), flags=re.I)

    # Replace all symbols
    words_all = re.sub(r"[^a-zA-Z0-9]", " ", words_all, flags=re.I)

    # Stem words
    words_all = [PorterStemmer().stem(word) for word in words_all]

    # Lemmatize words
    words_all_new = [WordNetLemmatizer().lemmatize(word) for word in words_all]

    return words_all_new


# Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)


# Parse argument for custom classifier
classifier = sys.argv[1]


# Load model
def load_model(classifier):
    """Instantiates a machine learning pipeline with a pickled classifier.

    :param classifier: str

    :return: sklearn.pipeline.Pipeline

    >>> model = load_model(ada_classifier.pkl)
    """

    model = joblib.load("models/{}".format(''.join(classifier)))

    return model


# Instantiate model with parsed custom classifier
model = load_model(classifier)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Data of original visualization
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Whether messages are related to disasters or not
    related_messages = df["related"].value_counts()
    related_messages_labels = ['related messages', 'not related messages']

    # Themes of Twitter messages
    df_temp = df.rename(columns={'aid_related':'aid',
                        'infrastructure_related':'infrastructure',
                        'weather_related':'weather'})
    themes = df_temp[['aid', 'infrastructure', 'weather']].sum()
    themes = themes.sort_values(ascending=False)
    themes_labels = list(themes.index)

    # Topics of themes of Twitter messages
    col_names = list(df.columns)
    topic_list = col_names[7:22] + col_names[23:31] + col_names[32:38]
    topics = df[topic_list].sum()
    topics = topics.sort_values(ascending=False)
    topics_labels = list(topics.index)

    # Create visuals
    graphs = [
        {
            'data' : [
                Pie(labels=genre_names,
                    values=genre_counts,
                    textinfo='percent',
                    textposition='outside',
                    hole=0.3)
                ],
            'layout' : {
                'title' : 'Source of messages',
                'titlefont' : {
                            'size' : '24'
                            }
                    }
        },
        {
            'data' : [
                Pie(labels=related_messages_labels,
                    values=related_messages,
                    textinfo='percent',
                    textposition='outside',
                    hole=0.3)
                ],
            'layout' : {
                'title' : 'Relevance of messages',
                'titlefont' : {
                            'size' : '24'
                            }
                    }
        },
        {
            'data' : [
                Pie(labels=themes_labels,
                    values=themes,
                    textinfo='percent',
                    textposition='outside',
                    hole=0.3)
                ],
            'layout' : {
                'title' : 'Themes of messages',
                'titlefont' : {
                            'size' : '24'
                            }
                    }
        },
        {
            'data' : [
                Bar(x=topics_labels,
                    y=topics)
                ],
            'layout' : {
                'title' : 'Topics of messages',
                'titlefont' : {
                            'size' : '24'
                            },
                'xaxis' : {
                        'title' : 'Topics',
                        'tickangle' : '-30',
                        'tickfont' : {
                                    'size' : '9'
                                    }
                        },
                'yaxis' : {
                        'title' : 'Count'
                        }
                    }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')
    app.logger.info('Text to be classified: {}'.format(query))

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    app.logger.info('Classification into: {}'.format(classification_labels))
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)


def main():
    print("Using the classifier saved in", ''.join(classifier), "to analyze new data.")
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
