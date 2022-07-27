import torch
from torch import nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, classes, hidden_dim, L):
        """
        classes: 9 ECG signals
        hidden_dim: 100
        L: ECG signal length 
        """
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(classes, hidden_dim)

        self.init_size = L // 4  # Initial size before upsampling, 2 upsampling due to // 4
        self.l1 = nn.Sequential(nn.Linear(hidden_dim, self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(1, 16, 3, stride=1, padding=1), # Input feature depends on the samples of labels
            # nn.Conv1d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Conv1d(16, 16, 3, stride=1, padding=1),
            nn.Conv1d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Conv1d(16, 16, 3, stride=1, padding=1),
            # nn.Conv1d(16, 16, 3, stride=1, padding=1),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Conv1d(32, 32, 3, stride=1, padding=1),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            nn.Upsample(scale_factor=2),
            # nn.Conv1d(32, 32, 3, stride=1, padding=1),
            nn.Conv1d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            # nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.Conv1d(64, 1, 3, stride=1, padding=1),
            # nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        noise: (batch_size, hidden_dim) 2D shape
        labels: (batch_size,) 1D shape
        """
        gen_input = torch.mul(self.label_emb(labels), noise) # dot product, shape: (batch_size, hidden_dim)
        out = self.l1(gen_input) # out shape: (batch_size, L // 4)
        out = out.unsqueeze(1) # out shape: (batch_size, 1, L // 4)
        ecg = self.conv_blocks(out) #  shape: (batch_size, 1, L)
        return ecg

# The second Generator
class Generator_2(nn.Module):
    def __init__(self, classes, hidden_dim, L):
        """
        hidden_dim: which is not used for below
        """
        super(Generator_2, self).__init__()

        self.label_emb = nn.Embedding(classes, L)

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            #
            nn.Conv1d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, padding= 1),
            #
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # 
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, padding= 1),
            # 
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            #
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),# Encoder part
            nn.MaxPool1d(3, padding= 1),
            #
            nn.ConvTranspose1d(128, 64, 3, 3, 1, output_padding= 2), # Decoder part
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            #
            nn.ConvTranspose1d(64, 32, 3, 3, 1), # Decoder part
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            # 
            nn.ConvTranspose1d(32, 1, 3, 3, 1), # Decoder part
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            # Output part
            nn.Sigmoid()
        )

    def forward(self, original, noise, labels):
        """
        original: (batch_size, 1, L) Original ECG signal.
        noise: (batch_size, L) 2D shape
        labels: (batch_size,) 1D shape
        """
        labels_embed = self.label_emb(labels)
        labels_embed = labels_embed.unsqueeze(1)
        noise = noise.unsqueeze(1)
        gen_input = torch.cat((original, noise, labels_embed), dim= 1) # Pix2Pix Generator
        ecg = self.conv_blocks(gen_input) #  shape: (batch_size, 1, L)
        return ecg

# The third Generator
class Generator_3(nn.Module):
    def __init__(self, classes, hidden_dim, L):
        """
        hidden_dim: which is not used for below
        """
        super(Generator_3, self).__init__()

        self.label_emb = nn.Embedding(classes, L)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            #
            nn.Conv1d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, padding= 1),
            #
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # 
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, padding= 1),
            # 
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            #
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, padding= 1)
        )# Encoder part

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, 3, 1, output_padding= 2), # Decoder part
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            #
            nn.ConvTranspose1d(64, 32, 3, 3, 1), # Decoder part
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            # 
            nn.ConvTranspose1d(32, 1, 3, 3, 1), # Decoder part
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            # Output part
            nn.Sigmoid()
        )

    def forward(self, original, noise, labels):
        """
        original: (batch_size, 1, L) Original ECG signal.
        noise: (batch_size, L) 2D shape
        labels: (batch_size,) 1D shape
        """
        labels_embed = self.label_emb(labels).unsqueeze(1) # (batch_size, 1, L)
        noise = noise.unsqueeze(1) # (batch_size, 1, L)
        gen_input = torch.mul(labels_embed, noise) # (batch_size, 1, L)

        ec_gen = self.encoder(gen_input) # encode for gen_input
        ec_gen_copy = ec_gen.clone() # For comparison
        ec_real = self.encoder(original) # encode for real samples
        ec_real_copy = ec_real.clone() # For comparison
        dc_gen = self.decoder(ec_gen) #  shape: (batch_size, 1, L) decode part for gen_input
        # dc_real = self.decoder(ec_real) #  shape: (batch_size, 1, L)

        return dc_gen, original, ec_gen_copy, ec_real_copy

# The LSTM Generator
class Generator_4(nn.Module):
    def __init__(self, classes, hidden_dim, L):
        """
        classes: 9 ECG signals
        hidden_dim: 100
        L: ECG signal length 
        """
        super(Generator_4, self).__init__()

        self.label_emb = nn.Embedding(classes, hidden_dim)

        self.init_size = L // 4  # Initial size before upsampling, 2 upsampling due to // 4
        self.l1 = nn.Sequential(nn.Linear(hidden_dim, self.init_size))
        HIDDEN_SIZE = 1

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(1, 16, 3, stride=1, padding=1), # Input feature depends on the samples of labels
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Conv1d(64, 64, 3, stride=1, padding=1),
            # nn.Conv1d(64, 1, 3, stride=1, padding=1),
        )
        self.rnn_layer = nn.LSTM(
                input_size= 64,
                hidden_size= HIDDEN_SIZE,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        self.out_layer = nn.Linear(2 * HIDDEN_SIZE, 1)

    def forward(self, noise, labels):
        """
        noise: (batch_size, hidden_dim) 2D shape
        labels: (batch_size,) 1D shape
        """
        self.rnn_layer.flatten_parameters() # To compact weights
        gen_input = torch.mul(self.label_emb(labels), noise) # dot product, shape: (batch_size, hidden_dim)
        out = self.l1(gen_input) # out shape: (batch_size, L // 4)
        out = out.unsqueeze(1) # out shape: (batch_size, 1, L // 4)
        out = self.conv_blocks(out).permute(0, 2, 1) #  shape: (batch_size, L, 64)
        out, _ = self.rnn_layer(out) #  shape: (batch_size, L, 2)
        ecg = self.out_layer(out).permute(0, 2, 1) #  shape: (batch_size, 1, L)

        return ecg

if __name__ == '__main__':

    L = 5500

    device = torch.device("cuda:0")
    generator = Generator_3(9, 500, L).to(device)

    z = torch.tensor(np.random.normal(0, 1, (5, L)), dtype= torch.float).to(device)

    # gen_one_hot = torch.eye(9).type(torch.long)
    gen_labels = torch.tensor(np.random.randint(0, 9, 5), dtype= torch.long).to(device)

    with torch.no_grad():
        ecg_out = generator(z.unsqueeze(1), z, gen_labels)
        print(ecg_out.shape)