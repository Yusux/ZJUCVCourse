import torch
from torchvision.transforms.functional import center_crop
from torch import nn


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    Crop the feature map from the contracting path to the size of the current feature map
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        b, c, h, w = x.shape

        # TODO: Concatenate the feature maps
        # use torchvision.transforms.functional.center_crop(...)
        x = torch.cat(
            (x, center_crop(contracting_x, (h, w))),
            dim=1
        )
        
        return x


class UNet(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # TODO: Double convolution layers for the contracting path.
        # Number of features gets doubled at each step starting from 64.        
        num = 64
        self.down_conv1 = nn.Sequential(
            # First convolution, turn in_channels to num
            # in_channels*H*W -> num*H*W
            nn.Conv2d(in_channels, num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num to num
            # num*H*W -> num*H*W
            nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down_conv2 = nn.Sequential(
            # First convolution, turn num to num*2
            # num*H*W -> (num*2)*H*W
            # > After MaxPool2d, H and W are halved
            # > For convenience, we note the halved H and W still as H and W
            nn.Conv2d(num, num * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num*2 to num*2
            # (num*2)*H*W -> (num*2)*H*W
            nn.Conv2d(num * 2, num * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down_conv3 = nn.Sequential(
            # First convolution, turn num*2 to num*4
            # (num*2)*H*W -> (num*4)*H*W
            nn.Conv2d(num * 2, num * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num*4 to num*4
            # (num*4)*H*W -> (num*4)*H*W
            nn.Conv2d(num * 4, num * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down_conv4 = nn.Sequential(
            # First convolution, turn num*4 to num*8
            # (num*4)*H*W -> (num*8)*H*W
            nn.Conv2d(num * 4, num * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num*8 to num*8
            # (num*8)*H*W -> (num*8)*H*W
            nn.Conv2d(num * 8, num * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Down sampling layers for the contracting path
        self.down_sample1 = nn.MaxPool2d(2)
        self.down_sample2 = nn.MaxPool2d(2)
        self.down_sample3 = nn.MaxPool2d(2)
        self.down_sample4 = nn.MaxPool2d(2)

        # TODO: The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = nn.Sequential(
            # First convolution, turn num*8 to num*16
            # (num*8)*H*W -> (num*16)*H*W
            nn.Conv2d(num * 8, num * 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num*16 to num*16
            # (num*16)*H*W -> (num*16)*H*W
            nn.Conv2d(num * 16, num * 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        # Set num to the number of features at the bottom of the U.
        num = 1024  
        # Pooling is using kernel_size=2, stride=2
        # Therefore, up-sampling is using kernel_size=2, stride=2
        self.up_sample1 = nn.ConvTranspose2d(num, num // 2, kernel_size=2, stride=2)
        self.up_sample2 = nn.ConvTranspose2d(num // 2, num // 4, kernel_size=2, stride=2)
        self.up_sample3 = nn.ConvTranspose2d(num // 4, num // 8, kernel_size=2, stride=2)
        self.up_sample4 = nn.ConvTranspose2d(num // 8, num // 16, kernel_size=2, stride=2)

        # TODO: Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the contracting path.
        # Therefore, the number of input features is double the number of features from up-sampling.
        self.up_conv1 = nn.Sequential(
            # First convolution, turn num to num//2
            # num*H*W -> (num//2)*H*W
            nn.Conv2d(num, num // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num//2 to num//2
            # (num//2)*H*W -> (num//2)*H*W
            nn.Conv2d(num // 2, num // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.Sequential(
            # First convolution, turn num//2 to num//4
            # (num//2)*H*W -> (num//4)*H*W
            nn.Conv2d(num // 2, num // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num//4 to num//4
            # (num//4)*H*W -> (num//4)*H*W
            nn.Conv2d(num // 4, num // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv3 = nn.Sequential(
            # First convolution, turn num//4 to num//8
            # (num//4)*H*W -> (num//8)*H*W
            nn.Conv2d(num // 4, num // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num//8 to num//8
            # (num//8)*H*W -> (num//8)*H*W
            nn.Conv2d(num // 8, num // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_conv4 = nn.Sequential(
            # First convolution, turn num//8 to num//16
            # (num//8)*H*W -> (num//16)*H*W
            nn.Conv2d(num // 8, num // 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution, turn num//16 to num//16
            # (num//16)*H*W -> (num//16)*H*W
            nn.Conv2d(num // 16, num // 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Crop and concatenate layers for the expansive path.
        # TODO: Implement class CropAndConcat starting from line 6
        self.concat1 = CropAndConcat()
        self.concat2 = CropAndConcat()
        self.concat3 = CropAndConcat()
        self.concat4 = CropAndConcat()
        
        # TODO: Final 1*1 convolution layer to produce the output
        self.final_conv = nn.Conv2d(num // 16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # TODO: Contracting path
        # Remember to pass middle result to the expansive path
        # 1st contracting path block
        contracting_x1 = self.down_conv1(x)
        x = self.down_sample1(contracting_x1)
        # 2nd contracting path block
        contracting_x2 = self.down_conv2(x)
        x = self.down_sample2(contracting_x2)
        # 3rd contracting path block
        contracting_x3 = self.down_conv3(x)
        x = self.down_sample3(contracting_x3)
        # 4th contracting path block
        contracting_x4 = self.down_conv4(x)
        x = self.down_sample4(contracting_x4)

        # Two 3*3 convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # TODO: Expansive path
        # Remember to receive from contracting path and concat
        # 1st expansive path block
        x = self.up_sample1(x)
        x = self.concat1(x, contracting_x4)
        x = self.up_conv1(x)
        # 2nd expansive path block
        x = self.up_sample2(x)
        x = self.concat2(x, contracting_x3)
        x = self.up_conv2(x)
        # 3rd expansive path block
        x = self.up_sample3(x)
        x = self.concat3(x, contracting_x2)
        x = self.up_conv3(x)
        # 4th expansive path block
        x = self.up_sample4(x)
        x = self.concat4(x, contracting_x1)
        x = self.up_conv4(x)

        # Final 1*1 convolution layer
        x = self.final_conv(x)

        return x