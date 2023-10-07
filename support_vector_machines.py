from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

training_data = 'data/train.txt'
output_filename = "data/svm_output.txt"
test_filename = 'data/unlabeled_test_test.txt'


def get_features(index, sentence):
    return {
        'word': sentence[index],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'bool_first_letter_capital': sentence[index] == (sentence[index].title()),
        'bool_first_word': index == 0,
        'bool_last': index == len(sentence) - 1,
        'bool_numeric': sentence[index].isdigit()
    }


def train_model():
    # Load training data
    train_data = []
    with open(training_data, 'r') as file:
        sentence_tokens = []
        for line in file:
            line = line.strip()
            if line:
                token, pos_tag, _ = line.split()
                sentence_tokens.append((token, pos_tag))
            else:
                train_data.append(sentence_tokens)
                sentence_tokens = []

    # Extract features
    x_train = [[get_features(index, sentence) for index in range(len(sentence))] for sentence in
               [list(zip(*sentence))[0] for sentence in train_data]]
    x_train = [word for sentence in x_train for word in sentence]
    y_train = [word[1] for sentence in train_data for word in sentence]

    # Train model
    model = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        # ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LinearSVC(dual='auto'))
    ])

    # https://youtu.be/l2I8NycJMCY?si=_Wp4T-JWuZhxWtQp
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    model.fit(x_train, y_train)

    # Evaluate model
    accuracy = model.score(x_test, y_test)
    print("Accuracy of dev set using SVM: ", accuracy)
    return model


def predict_tag():
    global output_filename, test_filename
    # Load the selected model
    model = train_model()

    # Load test data
    test_data = []
    with open(test_filename, 'r') as file:
        tokens = []
        for line in file:
            line = line.strip()
            if line:
                tokens.append(line)
            else:
                test_data.append(tokens)
                tokens = []

    # Extract features
    x_test = [get_features(index, sentence) for sentence in test_data for index in range(len(sentence) - 1)]
    y_pred = [model.predict(sentence_features) for sentence_features in x_test]
    test_data = [word for sentence in test_data for word in sentence]

    # Write output to file
    with open(output_filename, 'w') as f:
        for token, tag in zip(test_data, y_pred):
            f.write(f"{token} {tag[0]}\n")
        f.write("\n")
    print('Output written to data/svm_output.txt')


predict_tag()
