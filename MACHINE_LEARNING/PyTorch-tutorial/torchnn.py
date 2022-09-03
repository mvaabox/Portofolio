# Building a Neural Network with PyTorch

# Import dependencies
from torch import nn, save, load
# Choose the Optimizer
from torch.optim import Adam
# Load dataset from PyTorch
from torch.utils.data import DataLoader
from torchvision import datasets
# Convert images to Tensors
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLu(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLu(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLu(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu') # If you have a GPU, replace it with clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(10): # Train for 10 epoch
        for batch in dataset:
            X,y = batch
            X,y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backpropagation
            opt.zero_grad()
            loss.apply_backward()
            opt.step()

        print(f"Epoch: {epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)