import argparse
from unet import UNet
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

def plot_img_and_mask(img, mask, filename):
    classes = mask.max()
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask')
    ax[1].imshow(mask == 0)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model.pth',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='infer.jpg',
                        help='Specify the input image')
    parser.add_argument('--output', '-o', default='output.jpg',
                        help='Specify the output image')
    parser.add_argument('--threshold', '-t', default=0.9, type=float,
                        help='Specify the segmentation threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model {args.model}')
    print(f'Using device {device}')

    model = UNet(in_channels=3, out_channels=1).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    print('Model loaded')

    # Open image
    img = Image.open(args.input)
    transform = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device, dtype=torch.float32)

    # Inference
    with torch.no_grad():
        mask = model(img_tensor)

    # Postprocessing
    mask = mask.cpu()
    mask = F.interpolate(
        mask,
        size=img.size[::-1],
        mode='bilinear',
        align_corners=False
    )
    # Thresholding
    mask = torch.sigmoid(mask) > args.threshold
    # Change to HWC format
    mask = mask.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    # Plot
    plot_img_and_mask(img, mask, args.output)