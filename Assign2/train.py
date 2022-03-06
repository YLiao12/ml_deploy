# Imports
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from joblib import dump

from pre_process import split

def logistic_regression_count_bigram(train, test):
    """Train a logistic regression classifier with count vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Logistic Regression model with bigram CountVectorize...')
    # Extract documents and labels.
    docs_train = train['text']
    labels_train = train['label']
    docs_test = test['text']
    labels_test = test['label']
    # Start up a Pipeline
    pipe = Pipeline([
        ('vec', CountVectorizer(ngram_range=(1,2))),
        ('log', LogisticRegression())
    ])
    # Train the model.
    pipe.fit(docs_train, labels_train)
    # Do prediction.
    y_pred = pipe.predict(docs_test)
    # Get report.
    print(classification_report(labels_test, y_pred))
    dump(pipe, "count_model.pkl")

def logistic_regression_tfidf_bigram(train, test):
    """Train a logistic regression classifier with count vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Logistic Regression model with bigram TfidfVectorizer...')
    # Extract documents and labels.
    docs_train = train['text']
    labels_train = train['label']
    docs_test = test['text']
    labels_test = test['label']
    # Start up a Pipeline
    pipe = Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1,2))),
        ('log', LogisticRegression())
    ])
    # Train the model.
    pipe.fit(docs_train, labels_train)
    # Do prediction.
    y_pred = pipe.predict(docs_test)
    # Get report.
    print(classification_report(labels_test, y_pred))
    dump(pipe, "tfidf_model.pkl")

if __name__ == '__main__':
    train, test = split('G:\\machine learning\\Assign2\\', True, 'G:\\machine learning\\Assign2\\')
    logistic_regression_count_bigram(train, test)
    logistic_regression_tfidf_bigram(train, test)
