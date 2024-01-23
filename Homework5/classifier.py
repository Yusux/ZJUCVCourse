import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import cnn_train_transform, vit_train_transform, cnn_test_transform, vit_test_transform, PokemonDataset, decode_from_bin, decode_type, list_to_str
from models import AlexNetClassifier, ReferenceClassifier, ViTClassifier, test_model

# Load the processed data
pokemon = pd.read_pickle("dataset/pokemon_processed.pkl")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="alex", help="alex, reference or vit")
parser.add_argument("--dir", type=str, default="dataset", help="dataset directory")
parser.add_argument("--leak", action="store_true", help="use leak dataset")
args = parser.parse_args()

# Define the transformation and some hyperparameters
# according to the model
if args.model == "vit":
    print("Using ViT")
    train_transform = vit_train_transform
    test_transform = vit_test_transform
    batch_size = 128
else:
    print("Using CNN")
    train_transform = cnn_train_transform
    test_transform = cnn_test_transform
    batch_size = 256

# Define train csv file
if args.leak:
    print("Using leak dataset")
    train_csv_file = args.dir + "/leak.csv"
else:
    print("Using train dataset")
    train_csv_file = args.dir + "/train.csv"

# Define the dataset
train_dataset = PokemonDataset(
    csv_file=train_csv_file,
    image_dir=args.dir+"/images",
    transform=train_transform
)

test_dataset = PokemonDataset(
    csv_file=args.dir+"/test.csv",
    image_dir=args.dir+"/images",
    transform=test_transform
)

# Define the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define other hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4

# Define the model
if args.model == "alex":
    model = AlexNetClassifier(num_classes=pokemon["TypeBin"].values[0].shape[0]).to(device)
elif args.model == "reference":
    model = ReferenceClassifier(num_classes=pokemon["TypeBin"].values[0].shape[0]).to(device)
else:
    model = ViTClassifier(num_classes=pokemon["TypeBin"].values[0].shape[0]).to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
num_epochs = 500
writer = SummaryWriter(f"runs/{args.model}")

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels.float())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        writer.add_scalar("Loss/train", loss.item(), epoch*len(train_dataloader)+i)

    # Calculate accuracy on test set
    model.eval()
    with torch.no_grad():
        correct_rate = test_model(model, test_dataloader, device)

    # Record accuracy
    writer.add_scalar("Loss/train", loss.item(), epoch)
    writer.add_scalar("Accuracy/test", correct_rate, epoch)
    
    # Print loss and accuracy
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {correct_rate:.4f}")

writer.flush()
writer.close()

# After training, save some results
# Plot some images in train dataset (first 9 images)
model.eval()
with torch.no_grad():
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for i, (image, label) in enumerate(train_dataset):
        if i >= 9:
            break
        ax = axes[i//3, i%3]
        ax.imshow(np.clip(image.permute(1, 2, 0), 0, 1))
        true_label = decode_from_bin(label)
        pred_label = decode_type(model.predict(image.to(device)))
        ax.set_title(f"True: {list_to_str(true_label)}\nPred: {list_to_str(pred_label)}")
plt.tight_layout()
plt.savefig(f"{args.model}_train_set.png")

# Plot some images in test dataset (first 9 images)
with torch.no_grad():
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for i, (image, label) in enumerate(test_dataset):
        if i >= 9:
            break
        ax = axes[i//3-3, i%3]
        ax.imshow(np.clip(image.permute(1, 2, 0), 0, 1))
        true_label = decode_from_bin(label)
        pred_label = decode_type(model.predict(image.to(device)))
        ax.set_title(f"True: {list_to_str(true_label)}\nPred: {list_to_str(pred_label)}")
plt.tight_layout()
plt.savefig(f"{args.model}_test_set.png")
