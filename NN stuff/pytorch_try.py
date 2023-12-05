import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
from itertools import product

class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.layers(x)
        return out

'''
# Loss function and optimizer
loss_function = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
'''


# Dataset
class MyDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Split dataset into training set, validation set and test set
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# hyperparameters grid
hidden_sizes = [10, 20, 30]
learing_rates = [0.001, 0.01, 0.1]
num_epochs = 100

# Grid search
for hidden_size, learning_rate in product(hidden_sizes, learing_rates):
    net = ThreeLayerNN(input_size, hidden_size, output_size)

    # Loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate) 

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = net(train_data)
        # compute loss
        loss = loss_function(outputs, train_labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    with torch.no_grad():
        val_outputs = net(val_data)
        val_loss = loss_function(outputs, val_labels)

    # Print results of current hyperparameters
    print(f'hidden_size: {hidden_size}, learning_rate: {learning_rate}, val_loss: {val_loss}')
    print(f'Validation loss: {val_loss.item():.4f}\n')



# Dataloader for training set and validation set
train_dataset = MyDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = MyDataset(val_data, val_labels)
