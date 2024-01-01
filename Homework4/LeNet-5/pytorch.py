import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# define Hyper-parameter
train_batch_size = 128
test_batch_size = 1024
num_epochs = 10
learning_rate = 5e-4

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform image to tensor
transform = transforms.Compose([
   transforms.Resize((32,32)),
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])
# mean and std come from https://www.kaggle.com/code/berrywell/calculating-mnist-inverted-mnist-mean-std/notebook

# load dataset
# creates dataset from train-images-idx3-ubyte
train_set = torchvision.datasets.MNIST(
    root="./dataset",
    train=True,
    transform=transform,
    target_transform=None,
    download=True
)
train_loader = DataLoader(
    train_set,
    batch_size=train_batch_size,
    shuffle=True
)

# creates dataset from t10k-images-idx3-ubyte
test_set = torchvision.datasets.MNIST(
    root="./dataset",
    train=False,
    transform=transform,
    target_transform=None,
    download=True
)
test_loader = DataLoader(
   test_set,
   batch_size=test_batch_size,
   shuffle=False
)

# define model
class LeNetModel(nn.Module):
    def __init__(self):
        super(LeNetModel, self).__init__()
        
        # conv: 1*32*32 -> 6*28*28
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        # pool: 6*28*28 -> 6*14*14
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # conv: 6*14*14 -> 16*10*10
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # pool: 16*10*10 -> 16*5*5
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        # full connection: 400 -> 120
        self.fc_1 = nn.Linear(in_features=400, out_features=120)
        # full connection: 120 -> 84
        self.fc_2 = nn.Linear(in_features=120, out_features=84)
        # full connection: 84 -> 10
        self.fc_3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        # use view to flatten: 16*5*5 -> 400
        x = x.view(-1, 400)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

# create model
model = LeNetModel()
model = model.to(device)

# use CrossEntropyLoss as Loss function
criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# use Adam as optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# use tensorboard
writer = SummaryWriter()

for epoch in range(num_epochs):
    # switch model to train model
    loss_total = 0
    model.train()
    for i, (batch, label) in enumerate(train_loader):
        # send batch and label to device
        batch = batch.to(device)
        label = label.to(device)
        # forward
        output = model(batch)

        # clear grad
        optimizer.zero_grad()
        # get loss
        loss = criterion(output, label)
        loss_total += loss
        writer.add_scalar("Loss/train", loss, epoch*len(train_loader)+i)

        # backward, calculate the grad
        loss.backward()
        # update model
        optimizer.step()
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for testid, (test_data, test_label) in enumerate(test_loader):
        test_data = test_data.to(device)
        test_label = test_label.to(device)

        if (testid == 0):
          x = test_data
          for name, module in model._modules.items():
            if name == "fc_1":
                x = x.view(-1, 400)
            x = module(x)
            if name in ["conv_1", "pool_1", "conv_2", "pool_2"]:
              for idx in range(x[0].shape[0]):
                writer.add_images(f"image-{epoch}-[{name}]/train", x[0][idx], idx, dataformats="HW")

        test_pre = model(test_data)
        prediction = test_pre.argmax(dim=1)
        total += test_label.shape[0]
        correct += (prediction == test_label).sum().item()
    writer.add_scalar("Valid/train", correct/total, epoch)
    writer.add_scalar("Loss_epoch/train", loss_total, epoch)
    print(f"Epoch [{epoch+1:>2}/{num_epochs}] || correct/total = {correct}/{total} || Acc_rate = {correct/total:.2%}")

writer.flush()
writer.close()