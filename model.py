import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Hyper Parameters
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.001

# MNIST dataset
train_dataset = dsets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = dsets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor()
)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# GRNN Model
class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # FIX: was hardcoded 10, now uses num_classes

    def forward(self, x):
        # Set initial hidden state — no Variable needed in modern PyTorch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


grnn_model = GRNN(input_size, hidden_size, num_layers, num_classes)

# Load the saved model weights
# FIX: use weights_only=True to avoid the deprecation/security warning in PyTorch >= 2.0
state_dict = torch.load('grnn_model.pth', map_location=torch.device('cpu'), weights_only=True)
grnn_model.load_state_dict(state_dict)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(grnn_model.parameters(), lr=learning_rate)

# Train the Model
grnn_model.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # FIX: removed deprecated torch.autograd.Variable wrappers
        images = images.view(-1, 28, 28)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = grnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], '
                f'Step [{i + 1}/{len(train_dataset) // batch_size}], '
                f'Loss: {loss.item():.4f}'  # FIX: loss.data is deprecated; use loss.item()
            )

# Test the Model
grnn_model.eval()
correct = 0
total = 0
with torch.no_grad():  # FIX: disables gradient computation during evaluation (faster + correct)
    for images, labels in test_loader:
        images = images.view(-1, 28, 28)
        outputs = grnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # FIX: .item() to get a plain Python int

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# Save the Model
torch.save(grnn_model.state_dict(), 'grnn_model.pth')
