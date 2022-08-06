import torch
from torch import nn
import numpy as np
import sys
# from new_encoder import audio_encoder

# Using STA_CRNN as an anchor
PATH = '/home/alien/XUEYu/paper_code/enviroment1'
sys.path.append(PATH)
from STA_CRNN.cpsc2018_torch import vgg, Read_data

# Hyperparameters: seq_len, timestep
# Feature extractor
class G_enc(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, conv_arch, hidden_gru, gru_layers= 1):
        """
        conv_arch: List. Configure parameters for vgg
        timestep: int. < 63
        hidden_gru: int. Hidden dim of GRU
        gru_layers: the number of layers in GRU.
        """
        super(G_enc, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = vgg(conv_arch,12) # Tensor. (12, 15360) -> (256, 63) original
        # self.encoder =  audio_encoder() # Tensor. (12, 15360) -> (512, 96) original
        kwargs = dict(num_layers= gru_layers)  \
                if gru_layers == 1 else dict(num_layers= gru_layers, dropout= 0.2)
        self.gru = nn.GRU(256, hidden_gru, bidirectional=False, batch_first=True, **kwargs)
        # self.gru = nn.GRU(512, hidden_gru, bidirectional=False, batch_first=True, **kwargs) # For audio encoder
        self.Wk  = nn.ModuleList([nn.Linear(hidden_gru, 256) \
                                  for _ in range(timestep)])
        self.softmax  = nn.Softmax(dim= 1) # The dim of change is 0 
        self.lsoftmax = nn.LogSoftmax(dim= 1)
        self.hidden_gru = hidden_gru
        self.gru_layers = gru_layers

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):

        return torch.zeros(1 * self.gru_layers, batch_size, self.hidden_gru).cuda() \
               if use_gpu else torch.zeros(1 * self.gru_layers, batch_size, self.hidden_gru)
        
    def forward(self, x, hidden):
        """
        x:  8 * 12 * 15360, 8 means batch size.
        hidden: Tensor. dim(1, N, 128) 1: num_layers * bidirection. N: batch_size
        """
        batch = x.shape[0] # N. input sequence is N*C*L, e.g. 8 * 12 * 15360, 8 means batch size.
        
        z = self.encoder(x) # encoded sequence is N*C*L, e.g. 8 * 256 * 63
        t_samples = torch.randint(z.shape[-1] - self.timestep, size=(1,)).long() # [0, z.shape[-1]-timestep)
        z = z.transpose(1,2) # reshape to N*L*C for GRU, e.g. 8 * 63 * 256
        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, z.shape[-1])).float() # e.g. size 12*8*256

        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch, z.shape[-1]) # z_tk e.g. size 8*256
        
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8 * t_samples * 256
        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8 * t_samples * 128
        c_t = output[:,t_samples,:].view(batch, output.shape[-1]) # Last hidden dim. c_t e.g. size 8 * 128

        pred = torch.empty((self.timestep, batch, z.shape[-1])).float() # e.g. size 12 * 8 * 256
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8* 256
        
        accuracies = []
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8. diagonal line is the most similar.
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            accuracies.append(1.*float(f'{correct.item():.4f}')/batch)
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
            # returns a 1-D tensor with the diagonal elements of lsoftmax(total)
        
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden, accuracies

    def predict(self, x, hidden):
        """
        x : batch * 12 * 15360
        hidden: 1 * batch * hidden_gru
        """
        batch = x.size()[0]
        
        z = self.encoder(x) # batch * 256 * 63
        z = z.transpose(1,2) # batch * 63 * 256
        output, hidden = self.gru(z, hidden) # output size [(batch, 63, hidden_gru), 
                                             # (1 * batch  * hidden_gru)]

        return output, hidden # return every frame


class G_enc_original(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, conv_arch, hidden_gru):
        """
        conv_arch: List. Configure parameters for vgg
        timestep: int. < 63
        hidden_gru: int. Hidden dim of GRU
        gru_layers: the number of layers in GRU.
        """
        super(G_enc_original, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = vgg(conv_arch,12) # Tensor. (12, 15360) -> (256, 63) original
        self.gru = nn.GRU(256, hidden_gru, num_layers= 1, bidirectional=False, 
                            batch_first=True)
        
        self.Wk  = nn.ModuleList([nn.Linear(hidden_gru, 256) \
                                  for _ in range(timestep)])
        self.softmax  = nn.Softmax(dim= 1) # The dim of change is 0 
        self.lsoftmax = nn.LogSoftmax(dim= 1)
        self.hidden_gru = hidden_gru

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):

        return torch.zeros(1, batch_size, self.hidden_gru).cuda() \
               if use_gpu else torch.zeros(1, batch_size, self.hidden_gru)
        
    def forward(self, x, hidden):
        """
        x:  8 * 12 * 15360, 8 means batch size.
        hidden: Tensor. dim(1, N, 128) 1: num_layers * bidirection. N: batch_size
        """
        batch = x.shape[0] # N. input sequence is N*C*L, e.g. 8 * 12 * 15360, 8 means batch size.
        
        z = self.encoder(x) # encoded sequence is N*C*L, e.g. 8 * 256 * 63
        t_samples = torch.randint(z.shape[-1] - self.timestep, size=(1,)).long() # [0, z.shape[-1]-timestep)
        z = z.transpose(1,2) # reshape to N*L*C for GRU, e.g. 8 * 63 * 256
        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, z.shape[-1])).float() # e.g. size 12*8*256

        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch, z.shape[-1]) # z_tk e.g. size 8*256
        
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8 * t_samples * 256
        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8 * t_samples * 128
        c_t = output[:,t_samples,:].view(batch, output.shape[-1]) # Last hidden dim. c_t e.g. size 8 * 128

        pred = torch.empty((self.timestep, batch, z.shape[-1])).float() # e.g. size 12 * 8 * 256
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8* 256
        
        accuracies = []
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8. diagonal line is the most similar.
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            accuracies.append(1.*float(f'{correct.item():.4f}')/batch)
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
            # returns a 1-D tensor with the diagonal elements of lsoftmax(total)
        
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden, accuracies

    def predict(self, x, hidden):
        """
        x : batch * 12 * 15360
        hidden: 1 * batch * hidden_gru
        """
        batch = x.size()[0]
        
        z = self.encoder(x) # batch * 256 * 63
        z = z.transpose(1,2) # batch * 63 * 256
        output, hidden = self.gru(z, hidden) # output size [(batch, 63, hidden_gru), 
                                             # (1 * batch  * hidden_gru)]

        return output, hidden # return every frame


# classifier
class STA_Classifier(nn.Module): # Classifier 1
    def __init__(self, hidden_gru= 12):
        super().__init__()
        self.bidirectional = False
        # self.gru = nn.GRU(256, hidden_gru, num_layers=1, \
                         # dropout= 0.2,bidirectional= self.bidirectional, batch_first=True)
        self.gru = nn.GRU(256, hidden_gru, num_layers=1, \
                         bidirectional= self.bidirectional, batch_first=True)
        self.max_layer = nn.MaxPool1d(63)
        self.linear_layer = nn.Linear(2 * hidden_gru, 9) if self.bidirectional \
                            else nn.Linear(hidden_gru, 9)
    
    def forward(self, x):
        """
        x: Tensor. (B, F, T) B: batch size. F: the feature map. T: the time steps.
        """
        x = x.transpose(1, 2) # (B, T, F)
        x, _ = self.gru(x) # (B, T, 2 * H)
        x = x.transpose(1, 2) # (B, 2 * H,  T)
        x = self.max_layer(x) # (B, 2 * H,  1)
        x = x.flatten(1, -1) # (B, 2 * H)
        output = self.linear_layer(x) # (B, 9)

        return output


class ECG_Classifier(nn.Module): # Classifier 2
    ''' linear classifier '''
    def __init__(self, types, hidden_gru, droprate= 0.2):
        """
        types: int. ECG types: 9.
        hidden_gru: The number of hidden GRU units.
        droprate: float.
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_gru * 63, hidden_gru * 63),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(hidden_gru * 63, types)
        )

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x):
        # x: shape (batch, Hidden * T)
        x = self.classifier(x) # shape (batch, 9)

        return x


class Avg_Classifier(nn.Module): # Classifier_3
    def __init__(self, hidden_gru= 12):
        super().__init__()
        self.bn = nn.BatchNorm1d(2 * hidden_gru)
        self.ru = nn.ReLU()
        self.gru = nn.GRU(256, hidden_gru, num_layers=2, \
                         dropout= 0.2,bidirectional=True, batch_first=True)
        
        self.avg_layer = nn.AdaptiveAvgPool1d(1)
        self.linear_layer = nn.Linear(2 * hidden_gru, 9)
    
    def forward(self, x):
        """
        x: Tensor. (B, F, T) B: batch size. F: the feature map. T: the time steps.
        """
        x = x.transpose(1, 2) # (B, T, F)
        x, _ = self.gru(x) # (B, T, 2 * H)
        x = x.transpose(1, 2) # (B, 2 * H,  T)
        x = self.bn(x) # (B, 2 * H,  T)
        x = self.ru(x) # (B, 2 * H,  T)
        x = self.avg_layer(x) # (B, 2 * H,  1)
        x = x.flatten(1, -1) # (B, 2 * H)
        output = self.linear_layer(x) # (B, 9)

        return output


class Concat_Classifier(nn.Module): # Classifier_4 for GRU = 5
    def __init__(self, hidden_gru= 5, dropout= 0.2):
        super().__init__()

        self.max_layer = nn.MaxPool1d(63)
        self.avg_layer = nn.AdaptiveAvgPool1d(1)
        self.fc_head = nn.Sequential(
                       nn.Flatten(),
                       nn.Linear(3 * hidden_gru, 3 * hidden_gru),
                       nn.Dropout(dropout),
                       nn.Linear(3 * hidden_gru, 9)
                       )   
    def forward(self, x, hidden):
        """
        x: Tensor. (B, T, H) B: batch size. T: the time steps. H: the hidden dim of GRU.
        hidden: (1, batch, hidden_gru)
        """
        batch = x.shape[0] # Batch size
        x = x.transpose(1, 2) # (B, H, T)
        x_max = self.max_layer(x) # (B, H, 1)
        x_avg = self.avg_layer(x) # (B, H, 1)
        x_hidden = hidden.view(batch, -1, 1) # (B, H, 1)
        x_concat = torch.cat((x_max, x_avg, x_hidden), dim= 1) # (B, 3 * H, 1)
 
        output = self.fc_head(x_concat) # (B, 9)

        return output


if __name__ == '__main__':

    # Params
    device = torch.device('cuda:0')
    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]
    hidden_gru = 5
    params = (12, 2, 15360, conv_arch, hidden_gru)

    g_enc = G_enc(*params).to(device)

    input_data = torch.rand((2, 12, 15360)).to(device)
    input_hidden = torch.rand((1, 2, 128)).to(device)

    accuracy, nce, hidden = g_enc(input_data, input_hidden)