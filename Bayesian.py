# Write the code for bayesian classifier algo here
from idlelib.multicall import r

import numpy as np
import os
import math
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

# Declare filenames
training_data = 'train.txt'
test_data = 'test.txt'
output_file = 'pos_output.txt'


def read_train_data():
    pos_tokens = []
    pos_tags = []
    with open(training_data, 'r') as file:
        tokens = []
        tags = []
        for line in file:
            line = line.strip()
            if line:
                token, tag, _ = line.split()
                tokens.append(token)
                tags.append(tag)
            else:
                pos_tokens.append(tokens)
                pos_tags.append(tags)
                tags = []
                tokens = []
    return pos_tokens, pos_tags


def read_test_data():
    pos_tokens = []
    with open(test_data, 'r') as file:
        tokens = []
        for line in file:
            if line:
                tokens.append(line)
            else:
                pos_tokens.append(tokens)
                tokens = []
    return pos_tokens


def extract_features(sentence, index):
    word, pos = sentence[index]
    features = {
        'word': word,  # Current word
        'prev_word': '<s>',  # Default value for the previous word in a sentence
        'next_word': '</s>',  # Default value for the last word in a sentence
        'is_capitalized': word[0].isupper(),  # True if the word is capitalized
        'is_first_word': (index == 0),  # True if it's the first word in a sentence
        'is_last_word': (index == len(sentence) - 1),  # True if it's the last word in a sentence
    }

    # Update previous and next words if they exist
    if index > 0:
        features['prev_word'] = sentence[index - 1][0]
    if index < len(sentence) - 1:
        features['next_word'] = sentence[index + 1][0]

    return features


def train_bayesian_classifier(sentences):
    X, y = [], []
    for sentence in sentences:
        for i in range(len(sentence)):
            X.append(extract_features(sentence, i))
            X[-1] = extract_features(sentence, i)
            y.append(sentence[i][1])

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X)

    clf = MultinomialNB()
    clf.fit(X, y)
    return clf, vectorizer


def evaluate_classifier(classifier, vectorizer, test_sentences):
    X_test, y_test = [], []
    for sentence in test_sentences:
        for i in range(len(sentence)):
            X_test.append(extract_features(sentence, i))
            X_test[-1] = extract_features(sentence, i)
            y_test.append(sentence[i][1])

    X_test = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))


def predict_test_data(classifier, vectorizer, test_data, output_file):
    X_test = [extract_features(sentence) for sentence in test_data]
    X_test = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test)
    with open(output_file, 'w') as f:
        for sentence, pred in zip(test_data, y_pred):
            for word, tag in sentence:
                f.write(f"{word}\t{tag}\t{pred}\n")
            f.write("\n")
