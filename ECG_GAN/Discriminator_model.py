import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, L, classes):
        super(Discriminator, self).__init__()
        Feature_16 = 8

        def discriminator_block(in_channels, out_channels):
            """Returns layers of each discriminator block"""
            NUM_16 = 1
            NUM_32 = 0
            drop_rate = 0.2
            block = []

            for _ in range(NUM_16):
                block.append(nn.Conv1d(in_channels, Feature_16, 3, stride=1, padding=1))
                # block.append(nn.Conv1d(16, 16, 3, stride=1, padding=1))
                block.append(nn.BatchNorm1d(Feature_16))
                block.append(nn.LeakyReLU(0.2, inplace=True))
                block.append(nn.Dropout(drop_rate))
                in_channels = Feature_16

            for _ in range(NUM_32):
                block.append(nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1))
                # block.append(nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1))
                block.append(nn.BatchNorm1d(out_channels))
                block.append(nn.LeakyReLU(0.2, inplace=True))
                block.append(nn.Dropout(drop_rate))
                in_channels = out_channels
            
            # block.append(nn.Conv1d(out_channels, out_channels, 5, stride=1, padding=2))
            # block.append(nn.Conv1d(out_channels, out_channels, 5, stride=1, padding=2))

            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(1, 32) # Input 12-lead signal seperately 
        )

        ds_size = L # The length of ECG signal

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(Feature_16 * ds_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(Feature_16 * ds_size, classes))

    def forward(self, ecg):
        """
        ecg: shape (B, 1, L)
        """
        out = self.conv_blocks(ecg)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out) # (batch_size, 1)
        label = self.aux_layer(out) # (batch_size, classes)

        return validity, label