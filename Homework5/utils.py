import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Define the transformation
cnn_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.18961068, 0.1856215, 0.17901835,), 
        (0.35844862, 0.35119924, 0.34517958,)
    ),
    # The following lines are used to image augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20)
])

vit_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        antialias=True
    ),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ),
    # The following lines are used to image augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20)
])

cnn_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.18961068, 0.1856215, 0.17901835,), 
        (0.35844862, 0.35119924, 0.34517958,)
    )
])

vit_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Define the dataset
class PokemonDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        # Load the processed data
        pokemon = pd.read_pickle("dataset/pokemon_processed.pkl")
        self.pokemon_names = pd.read_csv(csv_file, header=None)[0]
        self.image_dir = image_dir
        self.transform = transform
        items = os.listdir(image_dir)
        self.image_names = [item for item in items if item.split(".")[0] in self.pokemon_names.values]
        # Load the images and labels into memory to accelerate training
        self.images = [np.array(Image.open(os.path.join(image_dir, image_name)).convert("RGBA"))[:, :, :3] for image_name in self.image_names]
        self.labels = [pokemon.loc[pokemon["Name"] == image_name.split(".")[0], "TypeBin"].values[0] for image_name in self.image_names]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Prepare image
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        # Prepare label
        label = self.labels[idx]
        
        return image, label

type_dict = {
    0: "Bug",
    1: "Dark",
    2: "Dragon",
    3: "Electric",
    4: "Fairy",
    5: "Fighting",
    6: "Fire",
    7: "Flying",
    8: "Ghost",
    9: "Grass",
    10: "Ground",
    11: "Ice",
    12: "Normal",
    13: "Poison",
    14: "Psychic",
    15: "Rock",
    16: "Steel",
    17: "Water"
}

def decode_type(indices):
    return [type_dict[idx] for idx in indices]

def bin_to_indices(bin):
    return np.where(bin == 1)[0].tolist()

def decode_from_bin(bin):
    return decode_type(bin_to_indices(bin))

def list_to_str(name_list):
    name_list.sort()
    return ", ".join(name_list)
