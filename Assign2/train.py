# Imports
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def naive_bayes_count(train, test):
    """Train a Naive Bayes classifier with count vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Naive Bayes model with unigram CountVectorize...')
    # Extract documents and labels.
    docs_train = train['comment']
    labels_train = train['attitude']
    docs_test = test['comment']
    labels_test = test['attitude']
    # Start up a Pipeline
    pipe = Pipeline([
        ('vec', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    # Train the model.
    pipe.fit(docs_train, labels_train)
    # Do prediction.
    y_pred = pipe.predict(docs_test)
    # Get report.
    print(classification_report(labels_test, y_pred))

def naive_bayes_tfidf(train, test):
    """Train a Naive Bayes classifier with Tf-Idf vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Naive Bayes model with unigram TfidfVectorize...')
    # Extract documents and labels.
    docs_train = train['comment']
    labels_train = train['attitude']
    docs_test = test['comment']
    labels_test = test['attitude']
    # Start up a Pipeline
    pipe = Pipeline([
        ('vec', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ])
    # Train the model.
    pipe.fit(docs_train, labels_train)
    # Do prediction.
    y_pred = pipe.predict(docs_test)
    # Get report.
    print(classification_report(labels_test, y_pred))

def logistic_regression_count(train, test):
    """Train a logistic regression classifier with count vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Logistic Regression model with unigram CountVectorize...')
    # Extract documents and labels.
    docs_train = train['comment']
    labels_train = train['attitude']
    docs_test = test['comment']
    labels_test = test['attitude']
    # Start up a Pipeline
    pipe = Pipeline([
        ('vec', CountVectorizer()),
        ('log', LogisticRegression())
    ])
    # Train the model.
    pipe.fit(docs_train, labels_train)
    # Do prediction.
    y_pred = pipe.predict(docs_test)
    # Get report.
    print(classification_report(labels_test, y_pred))

def logistic_regression_tfidf(train, test):
    """Train a logistic regression classifier with Tf-idf vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Logistic Regression model with unigram TfidfVectorize...')
    # Extract documents and labels.
    docs_train = train['comment']
    labels_train = train['attitude']
    docs_test = test['comment']
    labels_test = test['attitude']
    # Start up a Pipeline
    pipe = Pipeline([
        ('vec', TfidfVectorizer()),
        ('log', LogisticRegression())
    ])
    # Train the model.
    pipe.fit(docs_train, labels_train)
    # Do prediction.
    y_pred = pipe.predict(docs_test)
    # Get report.
    print(classification_report(labels_test, y_pred))

def logistic_regression_count_bigram(train, test):
    """Train a logistic regression classifier with count vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Logistic Regression model with biigram CountVectorize...')
    # Extract documents and labels.
    docs_train = train['comment']
    labels_train = train['attitude']
    docs_test = test['comment']
    labels_test = test['attitude']
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

def logistic_regression_tfidf_bigram(train, test):
    """Train a logistic regression classifier with count vectorizer.
    :param training set. pandas Dataframe.
    :param test set. pandas Dataframe.
    :param model save path. str. None for don't save.
    :return sklearn model.
    """
    print('Training Logistic Regression model with biigram CountVectorize...')
    # Extract documents and labels.
    docs_train = train['comment']
    labels_train = train['attitude']
    docs_test = test['comment']
    labels_test = test['attitude']
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

