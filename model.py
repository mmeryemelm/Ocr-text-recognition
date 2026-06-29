import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters 
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.001

# MNIST dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# GRNN Model
class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

grnn_model = GRNN(input_size, hidden_size, num_layers, num_classes)

# Load the saved Model
state_dict = torch.load('grnn_model.pth', map_location=torch.device('cpu'))
mapping = {'gru.weight_ih_l0': 'gru.weight_ih_l0', 'gru.weight_hh_l0': 'gru.weight_hh_l0', 'gru.bias_ih_l0': 'gru.bias_ih_l0', 'gru.bias_hh_l0': 'gru.bias_hh_l0', 'gru.weight_ih_l1': 'gru.weight_ih_l1', 'gru.weight_hh_l1': 'gru.weight_hh_l1', 'gru.bias_ih_l1': 'gru.bias_ih_l1', 'gru.bias_hh_l1': 'gru.bias_hh_l1', 'fc.weight': 'fc.weight', 'fc.bias': 'fc.bias'}
grnn_model.load_state_dict({mapping[k]: v for k, v in state_dict.items() if k in mapping})

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(grnn_model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28, 28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = grnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28, 28))
    outputs = grnn_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(grnn_model.state_dict(), 'grnn_model.pth')


