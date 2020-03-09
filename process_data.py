import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """Loads .csv files to the current working directory & merges them to a
    Pandas dataframe.

    Accepts paths to .csv files for messages & for categories as strings.

    :param messages_filepath: str
    :param categories_filepath: str

    :return: pandas.core.frame.DataFrame

    >>> df_categories, df = load_data('data/messages.csv', 'data/categories.csv')
    """

    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = df_messages.merge(df_categories, on="id")
    df.info()

    return df_categories, df

def clean_data(df_categories, df):
    """Extracts the information on disaster categories from a dataframe &
    replaces previous information on such categories in another dataframe.

    :param df_categories: pandas.core.frame.DataFrame
    :param df: pandas.core.frame.DataFrame

    :return: pandas.core.frame.DataFrame

    >>> df = clean_data(df_disaster_categories, df)
    """

    # Create & clean the information on disaster categories
    df_categories = df_categories.iloc[:, 1].str.split(';', expand=True)
    labels = df_categories.iloc[0, :]
    category_colnames = labels.replace(r'-[0-9]', '', regex=True).tolist()
    df_categories.columns = category_colnames
    df_categories.replace(r'[a-zA-Z]|_|-| ', '', inplace=True, regex=True)
    df_categories = df_categories.apply(pd.to_numeric)

    # Remove original messages & replace previous information on disaster categories
    df.drop(columns=['original', 'categories'], axis=1, inplace=True)
    df = pd.concat([df, df_categories], axis=1, ignore_index=False, sort=False)

    # Replace observations in disaster categories which are greater than 1,
    # assuming there were no greater values than 2
    check_unique_cats = df.iloc[:,4:].nunique()
    check_unique_cats_colnames = list(check_unique_cats[check_unique_cats > 2].index)
    for col in check_unique_cats_colnames:
        df[col].replace(2, 1, inplace=True)

    # Remove all entries with missing data on categories
    if len(df[df.isna().any(1)]) != 0:
        df.dropna(inplace=True)

    # Search for duplicated entries
    if len(df[df.duplicated()]) != 0:
        df.drop_duplicates(ignore_index=True, inplace=True)

    # Convert category columns with floats to integers
    df.iloc[:, 4:] = df.iloc[:, 4:].astype(int)
    df.info()

    return df

def save_data(df, database_filename):
    """Loads a Pandas dataframe to a new SQLite database.

    :param df: pandas.core.frame.DataFrame
    :param database_filename: str
    :param table_name: str

    :return: None

    >>> save_data(df, 'newsqlite.db')
    """

    connection = create_engine(''.join(['sqlite:///', database_filename]))
    df.to_sql('categorized_messages', connection, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_categories, df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df_categories, df)

        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              './data/disaster_messages.csv ./data/disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
