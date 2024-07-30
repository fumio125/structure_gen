import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_channels):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            DiscriminatorBlock(image_channels, hidden_channels),
            DiscriminatorBlock(hidden_channels, hidden_channels * 2),
            DiscriminatorBlock(hidden_channels * 2, 1, final_layer=True)
        )

    def forward(self, input_images):
        prediction = self.discriminator(input_images)
        return prediction.view(len(prediction), -1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        super(DiscriminatorBlock, self).__init__()
        if not final_layer:
            self.discriminator_block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            self.discriminator_block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, x):
        return self.discriminator_block(x) 