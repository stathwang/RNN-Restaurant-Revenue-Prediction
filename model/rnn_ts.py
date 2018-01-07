import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(888)

def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# model parameters
learning_rate = 0.01
num_epochs = 5000
input_size = 2
hidden_size = 2
num_classes = 1
timesteps = seq_length = 5
num_layers = 1

filedir = 'data/'
dat = pd.read_csv(filedir + 'restaurant_revenue_final.csv')

print(dat.head())

dat = dat[['credit', 'total']]
xy = np.asarray(dat)
xy = min_max_scaler(xy)

xy = xy[::-1]
x = xy
y = xy[:,[-1]]

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:(i+seq_length)]
    _y = y[i+seq_length]
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size
trainX = torch.Tensor(np.array(dataX[0:train_size]))
trainX = Variable(trainX)
testX = torch.Tensor(np.array(dataX[train_size:len(dataX)]))
testX = Variable(testX)
trainY = torch.Tensor(np.array(dataY[0:train_size]))
trainY = Variable(trainY)
testY = torch.Tensor(np.array(dataY[train_size:len(dataY)]))
testY = Variable(testY)

class LSTMTS(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTMTS, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size), requires_grad=True)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size), requires_grad=True)

        # Forward proprogate input through LSTM
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

# Instantiate LSTM model
lstm = LSTMTS(num_classes, input_size, hidden_size, num_layers)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # Obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    
    print("Epoch: %d', loss: %1.5f" % (epoch, loss.data[0]))

# Test the model
lstm.eval()
test_predict = lstm(testX)
train_predict = lstm(trainX)

# Plot predictions
test_predict = test_predict.data.numpy()
train_predict = train_predict.data.numpy()
testY = testY.data.numpy()
trainY = trainY.data.numpy()

plt.plot(testY)
plt.plot(test_predict)

#plt.plot(trainY)
#plt.plot(train_predict)
plt.xlabel('Time Period')
plt.ylabel('Restaurant Revenue')
plt.show()
