from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

output_filename = "data/svm_output.txt"
test_filename = 'data/unlabeled_test_test.txt'


def get_features(index, sentence):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'capital': sentence[index] == (sentence[index].title()),
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    }


def train_model(evaluate=False):
    train_data = []
    with open('data/train.txt', 'r') as file:
        sentence_tokens = []
        for line in file:
            line = line.strip()
            if line:
                token, pos_tag, _ = line.split()
                sentence_tokens.append((token, pos_tag))
            else:
                train_data.append(sentence_tokens)
                sentence_tokens = []
    x_train = [[get_features(index, sentence) for index in range(len(sentence))] for sentence in
               [list(zip(*sentence))[0] for sentence in train_data]]
    x_train = [word for sentence in x_train for word in sentence]
    y_train = [word[1] for sentence in train_data for word in sentence]

    model = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        # ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LinearSVC(C=0.1, dual='auto'))
    ])

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print("Accuracy of dev set using SVM: ", accuracy)
    if evaluate:
        return accuracy
    return model


def predict_tag(evaluate=False):
    global output_filename, test_filename
    # Load the selected model
    model = train_model(evaluate)

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

    x_test = [get_features(index, sentence) for sentence in test_data for index in range(len(sentence) - 1)]

    y_pred = [model.predict(sentence_features) for sentence_features in x_test]

    test_data = [word for sentence in test_data for word in sentence]

    with open(output_filename, 'w') as f:
        for token, tag in zip(test_data, y_pred):
            f.write(f"{token} {tag[0]}\n")
        f.write("\n")
    print(f"POS tagging completed! Results saved to {output_filename}")


predict_tag()
