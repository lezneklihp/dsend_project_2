from lightgbm import LGBMClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import re
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_database_data(database_filename):
    """Loads data from a SQLite database to a Pandas dataframe & splits the data
    into features and targets.

    :param database_filename: str

    :return: pandas.core.frame.DataFrame

    >>> df = load_database_data('cleaned_disaster_data.db')
    """

    connection = create_engine(''.join(['sqlite:///', database_filename]))
    df = pd.read_sql_query('SELECT * FROM categorized_messages', connection)
    X = df['message'].values
    y = df.iloc[:, 3:]
    category_names = list(y.columns)

    return X, y, category_names

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

def build_model():
    """Instantiates a machine learning pipeline with a AdaBoostClassifier.

    :return: sklearn.pipeline.Pipeline

    >>> model = build_model()
    """

    pipeline = make_pipeline(
                    TfidfVectorizer(
                                    tokenizer=tokenize,
                                    smooth_idf=False),
                    MultiOutputClassifier(
                                    AdaBoostClassifier(),
                    n_jobs=-1))

    parameters_dict = dict(multioutputclassifier__estimator__n_estimators=[20, 50, 200],
                       multioutputclassifier__estimator__learning_rate=[0.01, 0.2])

    model = GridSearchCV(pipeline, param_grid=parameters_dict, cv=2, verbose=3, n_jobs=-1)

    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    """Creates a report on how well the pipeline is able to categorize data.

    :param model: sklearn.pipeline.Pipeline
    :param X_test: numpy.ndarray
    :param y_test: pandas.core.frame.DataFrame
    :param category_names: list

    >>> evaluate_model(model, X_test, Y_test, category_names)
    """

    y_pred = model.predict(X_test)

    for col in range(len(category_names)):
        result = classification_report(y_test.iloc[:,col], y_pred[:,col])
        print("Report on", category_names[col], ":")
        print(result)
        print("F1-score of positive classes:", f1_score(y_test.iloc[:,col], y_pred[:,col], labels=np.unique(y_pred), average=None))
        print("F1-score (micro):", f1_score(y_test.iloc[:,col], y_pred[:,col], labels=np.unique(y_pred), average='micro'))
        print("F1-score (macro):", f1_score(y_test.iloc[:,col], y_pred[:,col], labels=np.unique(y_pred), average='macro'))
        print("F1-score (weighted):", f1_score(y_test.iloc[:,col], y_pred[:,col], labels=np.unique(y_pred), average='weighted'))
        print("Accuracy score:", accuracy_score(y_test.iloc[:,col], y_pred[:,col]))


def save_model(model, model_filepath):
    """Saves the current, trained pipeline to a pickle file.

    :param model: sklearn.pipeline.Pipeline
    :param model_filepath: str

    >>> evaluate_model(model, 'models/classifier.pkl')
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filename))
        X, y, category_names = load_database_data(database_filename)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

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
              'train_classifier.py data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()
