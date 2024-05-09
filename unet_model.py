
""" Full assembly of the parts to form the complete network """

from unet_parts import *
import torch
import torch.nn as nn
import torch.optim as optim


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,global_length,bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.global_length = global_length

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)

        # adding cbam during the downsampling of the unet
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)

        self.up2 = Up(512+global_length, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # adding cbam during the downsampling of the unet - Chenjia
        self.cbam4 = CBAM(channel=256 // factor)
        self.cbam5 = CBAM(channel=128 // factor)
        self.cbam6 = CBAM(channel=64)

        self.lstm_input_size = (512 // factor) * 4 * 4
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x, x_global):
        x1 = self.inc(x)    #output : 32*32*64
        x1 = self.cbam1(x1) + x1

        x2 = self.down1(x1)     #output : 16*16*128
        x2 = self.cbam2(x2) + x2

        x3 = self.down2(x2)     #output : 8*8*256
        x3 = self.cbam3(x3) + x3

        x4 = self.down3(x3)     #output : 4*4*512

        lstm_input = x4.view(x4.size(0), 1, self.lstm_input_size)
        lstm_output, _ = self.lstm(lstm_input)
        transformer_input = lstm_output.permute(1, 0, 2)
        transformer_output = self.transformer(transformer_input)
        x4 = torch.cat([x4, transformer_output.permute(1, 0, 2).squeeze(1)], dim=1)

        x = self.up2(x4, x3)
        x = self.cbam4(x) + x

        x = self.up3(x, x2)
        x = self.cbam5(x) + x

        x = self.up4(x, x1)
        x = self.cbam6(x) + x
        
        logits = self.outc(x)
        return logits
