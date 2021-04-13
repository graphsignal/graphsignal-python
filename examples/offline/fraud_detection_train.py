import pandas as pd
import numpy as np
import pickle

from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fraud_model import FraudModel

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# download from https://www.kaggle.com/mlg-ulb/creditcardfraud
df = pd.read_csv('../../test-data/fraud-detection/creditcard.csv')

df.head(5)
print(df.shape)
print(df.describe())

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0, color="g")

transactions = df.drop(['Time'], axis=1)
transactions['Amount'] = StandardScaler().fit_transform(transactions['Amount'].values.reshape(-1, 1))

X_train, X_test = train_test_split(transactions, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values
y_test = y_test.values

model = FraudModel().double().cpu()

num_epochs = 40
minibatch_size = 32
learning_rate = 1e-3

train_loader = DataLoader(X_train, batch_size=minibatch_size, shuffle=True)
test_loader = DataLoader(X_test, batch_size=1, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
model.parameters(), lr=learning_rate, weight_decay=10e-05)

history = {}
history['train_loss'] = []
history['test_loss'] = []

for epoch in range(num_epochs):
    h = np.array([])
    for data in train_loader:
        output = model(data)
        loss = criterion(output, data)
        h = np.append(h, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = np.mean(h)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, mean_loss))
    history['train_loss'].append(mean_loss)

torch.save(model.state_dict(), './fraud_model.pt')
