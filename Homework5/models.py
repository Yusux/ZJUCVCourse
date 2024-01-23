import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from utils import bin_to_indices

# Define Super Class of PokemonClassifier

class PokemonClassifier(nn.Module):
    def __init__(self):
        super(PokemonClassifier, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def predict(self, images):
        with torch.no_grad():
            # predict
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            output = self(images)
            # take the top 2
            _, indices = torch.topk(output, 2)
            return indices.squeeze().tolist()

# Alexnet model
# Input size: 120 * 120 * 3
class AlexNetClassifier(PokemonClassifier):
    def __init__(self, num_classes):
        super(PokemonClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2), # 55 * 55 * 96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27 * 27 * 96
            nn.Conv2d(96, 256, kernel_size=5, padding=1), # 25 * 25 * 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 12 * 12 * 256
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 12 * 12 * 384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 12 * 12 * 384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 12 * 12 * 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 5 * 5 * 256
        )

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(5 * 5 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# Reference model
# Input size: 120 * 120 * 3
class ReferenceClassifier(PokemonClassifier):
    def __init__(self, num_classes):
        super(PokemonClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),        # 118 * 118 * 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 59 * 59 * 16
            nn.Conv2d(16, 32, kernel_size=3),       # 57 * 57 * 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 * 28 * 32
            nn.Conv2d(32, 64, kernel_size=3),       # 26 * 26 * 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 13 * 13 * 64
            nn.Conv2d(64, 128, kernel_size=3),      # 11 * 11 * 128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5 * 5 * 128
            nn.Conv2d(128, 150, kernel_size=3),     # 3 * 3 * 150
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(150),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1 * 1 * 150
        )

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(150, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# ViT based model
class ViTClassifier(PokemonClassifier):
    def __init__(
        self,
        num_classes,
        config=ViTConfig(),
        model_checkpoint="google/vit-base-patch16-224-in21k"
    ):
        super(ViTClassifier, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, config=config)

        self.dense = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        x = self.vit(x)['last_hidden_state'][:, 0, :]
        x = self.dense(x)
        return x

def test_model(model, dataloader, device):
    num_correct = 0
    num_samples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        pred_indices = model.predict(images)
        label_indices = [bin_to_indices(label) for label in labels.cpu().numpy()]
        for (pred, label) in zip(pred_indices, label_indices):
            pred.sort()
            label.sort()
            if (len(label) == 1):
                if label[0] in pred:
                    num_correct += 1
            else:
                if label == pred:
                    num_correct += 1
        num_samples += labels.size(0)
    return num_correct / num_samples * 100
