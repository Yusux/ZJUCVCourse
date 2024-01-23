import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import os

pokemon = pd.read_csv("dataset/pokemon.csv")

# Type1 and Type2 unique and remove nan
types = pokemon[["Type1", "Type2"]].dropna().melt()["value"].unique()

mlb = MultiLabelBinarizer()
mlb.fit([types])

print("Classes:")
for i, label in enumerate(mlb.classes_):
    print(f"{i:2}: {label}")

# Add Type and TypeBin column
pokemon["Type"] = pokemon.apply(
    lambda row: [row["Type1"]] if pd.isnull(row["Type2"]) else [row["Type1"], row["Type2"]],
    axis=1
)
pokemon["TypeBin"] = pokemon["Type"].apply(lambda x: mlb.transform([x])[0])


# Calculate the mean and std value of pokemon images for transformation
image_dir = "dataset/images"
image_files = os.listdir(image_dir)

image_pixels = []

for image_file in image_files:
    image = Image.open(os.path.join(image_dir, image_file)).convert("RGBA")
    image = np.asarray(image)[:, :, :3]
    image_pixels.append(image)

image_mean = np.mean(image_pixels, axis=(0, 1, 2)) / 255
image_std = np.std(image_pixels, axis=(0, 1, 2)) / 255
print("Mean:", image_mean)
print("Std:", image_std)

# Delete Type column
pokemon_tosave = pokemon.drop("Type", axis=1)
# Save the processed data
pokemon_tosave.to_pickle("dataset/pokemon_processed.pkl")
