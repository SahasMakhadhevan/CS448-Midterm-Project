import torch
from torch import nn
#import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
#from sklearn import datasets
#from sklearn.preprocessing import StandardScaler    
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

output_filename = 'Logistic_Regression.txt'
input_filename = 'unlabeled_test_test.txt'

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

# Load training data
train_data = []
with open('train.txt', 'r') as file:
    sentence_tokens = []
    for line in file:
        line = line.strip()
        if line:
            token, pos_tag, _ = line.split()
            sentence_tokens.append((token, pos_tag))
        else:
            train_data.append(sentence_tokens)
            sentence_tokens = []

#def get_features(index, sentence):

#sentences = ["The cat sat on the mat.", "I like to play soccer."]
#pos_tags = ["DT NN VBD IN DT NN", "PRP VBP TO VB NN."]

"""
word_counter = Counter()
for sentence in sentences:
    word_counter.update(sentence.split())
vocab = {word: idx for idx, (word, _) in enumerate(word_counter.items())}

X = [[vocab[word] for word in sentence.split()] for sentence in sentences]
encoder = LabelEncoder()
y = encoder.fit_transform(pos_tags)
"""

# Extract features
x_train = [[get_features(index, sentence) for index in range(len(sentence))] for sentence in
            [list(zip(*sentence))[0] for sentence in train_data]]
x_train = [word for sentence in x_train for word in sentence]
y_train = [word[1] for sentence in train_data for word in sentence]

"""
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape
print(n_samples, n_features)
"""

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

"""
#scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
"""

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size = 32, shuffle=True)



#Model

"""
#f = wx + b, sigmoid at the end
class LogisticRegression(nn.module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
"""

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
input_dim = len(x_train)
output_dim = len(y_train)
model = LogisticRegression(input_dim, output_dim)

#model = LogisticRegression(n_features)

#Loss and Optimizer
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)

accuracy = (predicted == y_test).sum().item()/y_test.size(0)
print(f"Accuracy: {accuracy}")

"""
#Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    #backward pass
    loss.backward()

    #updates
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if(epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
"""
