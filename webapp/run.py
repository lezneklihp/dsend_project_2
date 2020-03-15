from flask import Flask
from flask import render_template, request, jsonify
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import plotly
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens

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

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)

classifier = sys.argv[1]

# load model
def load_model(classifier):
    """Instantiates a machine learning pipeline with a pickled classifier.

    :param classifier: str

    :return: sklearn.pipeline.Pipeline

    >>> model = load_model(ada_classifier.pkl)
    """

    model = joblib.load("models/{}".format(''.join(classifier)))

    return model

model = load_model(classifier)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    genre_aid_sum = df[['medical_help', 'medical_products',
       'search_and_rescue', 'security', 'military', 'child_alone', 'water',
       'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
       'death', 'other_aid', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather']].sum().sort_values(ascending=False)

    aid_names = list(genre_aid_sum.index)

    request_offer = df[["request", "offer"]].sum()
    request_offer_name = list(request_offer.index)

    related = df[["aid_related", "infrastructure_related", "weather_related"]].sum().sort_values(ascending=False)
    related_name = list(related.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
    print("Using the classifier saved in", ''.join(classifier), "to analyze new data.")
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
