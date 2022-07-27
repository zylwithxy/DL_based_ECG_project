import torch
from torch import nn
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from Generator_model import Generator # Conv model
from Generator_model import Generator_4 # LSTM model

class LoadData():
    def __init__(self, CLASSES, HIDDEN_DIM, L, PATH, SAMPLE_NUM, LABEL, MODEL_NAME= 'gen_4'):
        """
        sample_num: Number of samples generated.
        MODEL_NAME: str: "gen_1", 'gen_2', ...
        """
        self.CLASSES = CLASSES
        self.HIDDEN_DIM = HIDDEN_DIM
        self.L = L
        self.PATH = PATH
        self.SAMPLE_NUM = SAMPLE_NUM
        self.MODEL_NAME = MODEL_NAME

        if MODEL_NAME == "gen_1":
            self.MODEL = Generator
        elif MODEL_NAME == "gen_4":
            self.MODEL = Generator_4

        self.devices = None
        self.z, self.gen_labels = None, None
        self.generator = self.load_model()
        self.z, self.gen_labels = self.produce_input(LABEL)

        self.gen_results = self.generate_leads() # Training samples -> Tensor (SAMPLE_NUM, 12, F_l)
        self.gen_labels = self.transfer_multi_labels() # Training labels -> Tensor (SAMPLE_NUM, 3)

    def load_model(self):

        devices = [torch.device(f'cuda:{i}') for i in range(2)]
        self.devices = devices
        generator = self.MODEL(self.CLASSES, self.HIDDEN_DIM, self.L).to(devices[0])
        generator = nn.DataParallel(generator, device_ids=devices)
        return generator

    def produce_input(self, label):
        # label: the label of generated sample 

        assert type(label) == int
        z = torch.tensor(np.random.normal(0, 1, (self.SAMPLE_NUM, self.HIDDEN_DIM)), dtype= torch.float).to(self.devices[0])
        labels = np.array([label for _ in np.arange(self.SAMPLE_NUM)], dtype= np.long)
        gen_labels = torch.tensor(labels, dtype= torch.long).to(self.devices[0])

        return z, gen_labels

    def generate_leads(self):

        gen_results = []
        common_fname = "lead_model_weights_"

        for lead in range(12):

            filename = f'{lead}'+ common_fname + '1.pth' if self.MODEL_NAME == 'gen_4' \
                       else f'{lead}' + common_fname + f'{lead+1}.pth'
            generator_params = torch.load(op.join(self.PATH, filename))
            self.generator.load_state_dict(generator_params)
            self.generator.eval()
            with torch.no_grad():
                gen_lead = self.generator(self.z, self.gen_labels) # shape: (SAMPLE_NUM, 1, L)
                gen_lead = gen_lead.cpu()
            gen_results.append(gen_lead)
    
        gen_results = torch.cat(gen_results, axis= 1)
        
        return gen_results
    
    def transfer_multi_labels(self):
        """Because some ECG samples are multi-label"""

        temp = self.gen_labels.cpu().unsqueeze(1).numpy() # shape: (SAMPLE_NUM, 1) ndarray
        fill = np.array([np.nan for _ in np.arange(self.SAMPLE_NUM)]).reshape(self.SAMPLE_NUM, -1) # shape: (SAMPLE_NUM, 1) ndarray
        multi_labels = np.concatenate((temp, fill, fill), axis= 1) # shape (SAMPLE_NUM, 3)
        multi_labels = torch.tensor(multi_labels, dtype= torch.int64)

        return multi_labels


def plot_gen_lead(gen_lead, num):
        """
        gen_lead: shape(SAMPLE_NUM, 1, L)
        num: The number of samples showed
        """

        fig, axes = plt.subplots(num, sharex=True, 
                                 figsize=(15, 3*num),
                                    ) # 36
        axes = axes.ravel()

        for i in np.arange(num): # lead
            axes[i].grid(which='both', axis='both', linestyle='--')
            axes[i].plot(gen_lead[i][0], color="salmon", label= 'ECG signal')
            axes[i].set_title(f'Sample {i+1}')
            if i == 0:
                axes[i].legend(loc= 'upper right')
        
        fig.tight_layout()
        top, bottom = fig.subplotpars.top, fig.subplotpars.bottom
        fig.subplots_adjust(top= top - 0.025, bottom= bottom + 0.024, hspace=0.8)
        plt.show()


def plot_gen_samples(gen_result, sample, label):
    """
    gen_result: (SAMPLE_NUM, 12, L)
    sample: The index of SAMPLE_NUM
    label: int [0, 8]. The label of generated result.
    """
    fig, axes = plt.subplots(12, sharex=True, 
                             figsize=(15, 36),
                             ) # 36
    axes = axes.ravel()

    for i in np.arange(12): # lead
        axes[i].grid(which='both', axis='both', linestyle='--')
        axes[i].plot(gen_result[sample][i], color="salmon", label= 'ECG signal')
        axes[i].set_title(f'Label{label+1}_Lead{i+1}')
        if i == 0:
            axes[i].legend(loc= 'upper right')
        
    fig.tight_layout()
    top, bottom = fig.subplotpars.top, fig.subplotpars.bottom
    fig.subplots_adjust(top= top - 0.025, bottom= bottom + 0.024, hspace=0.8)
    plt.show()

if __name__ == '__main__':

    CLASSES, HIDDEN_DIM, L = 9, 400, 5500
    PATHs = ['/home/alien/XUEYu/paper_code/Parameters/2018_Generator',\
             '/home/alien/XUEYu/paper_code/Parameters/2018_Generator/Generator',
             '/home/alien/XUEYu/paper_code/Parameters/2018_Generator/Generator_rmv_tanh']
    
    SAMPLE_NUM = 10 # Number of samples generated
    LABEL = 0

    gen_leads = LoadData(CLASSES, HIDDEN_DIM, L, PATHs[-1], SAMPLE_NUM, LABEL, "gen_1")
    print(gen_leads.gen_results.shape)
    print(gen_leads.gen_labels.shape)
    print(gen_leads.gen_labels.dtype)
    print(gen_leads.gen_labels[0][0])
    plot_gen_samples(gen_leads.gen_results, 2, LABEL)