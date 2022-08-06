import torch
import torch.nn as nn
import sys

PATH = '/home/alien/XUEYu/paper_code/enviroment1'
sys.path.append(PATH)
from STA_CRNN.CBAM import CBAMBlock

def vgg_block(num_convs, in_channels, out_channels, ):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        in_channels = out_channels
        layers.append(nn.BatchNorm1d(in_channels))
        layers.append(nn.ReLU())

    layers.append(nn.MaxPool1d(3))
    layers.append(nn.Dropout(0.2))
    
    return nn.Sequential(*layers)

def vgg(conv_arch, in_channels):
    conv_blks = []
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        conv_blks.append(CBAMBlock(channel = in_channels))
        # CBAMBlock.init_weights()

    # Global feature extraction
    # conv_blks.append()
    conv_blks.pop() # 排除最后一个attention 22/03/14
    return nn.Sequential(
        *conv_blks)


def audio_encoder():
    """
    The encoder is used for audio classification.
    """
    encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(12, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
    return encoder

if __name__ == "__main__":
    
    encoder = audio_encoder()
    encoder = encoder.cuda()

    X = torch.rand(2, 12, 15360).cuda()
    output = encoder(X)
    print(output.shape)