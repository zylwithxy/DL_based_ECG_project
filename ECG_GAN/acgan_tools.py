import torch
import os
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

PATH = '/home/alien/XUEYu/paper_code/enviroment1/VAN'
sys.path.append(PATH)
from dataset import TraindataSet

class Configure_Data():

    def __init__(self, data, label, bs):
        """
        data: ndarray. (N, 12, 5500) Training data
        label: ndarray. (N,) 1-D Training labels
        bs: Batch size
        """
        temp_data = self.to_tensor_format(data, label)
        self.train_data = TraindataSet(*temp_data)
        self.dataloader = DataLoader(self.train_data, batch_size= bs, shuffle= True)

    def to_tensor_format(self, data, label):
        
        data = torch.FloatTensor(data)
        label = torch.LongTensor(label) - 1
        # print(torch.all(label <= 8))

        return (data, label)
    
    def output_dataloader(self):
        return self.dataloader

class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:', 'c-', 'y:'), nrows=3, ncols=1,
                 figsize=(7, 5)):
        """
        xlabel: str
        ylabel: list len(ylabel) == 3
        legend: list 2D shape. shape: (3, 2)
        xlim: [1, epoch]
        ylim: None
        """
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize, sharex= True)

        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim

        self.X, self.Y, self.fmts = None, None, fmts
        self.path = '/home/alien/XUEYu/Draft/GAN_pic'

    def add(self, x, y):
        # Add multiple data points into the figure
        """
        x: Epoch
        y: Tuple length n. 
        """
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
    
        for i in np.arange(len(self.axes)):
            self.axes[i].cla()
            self.axes[i].grid()

        for count, (x, y, fmt) in enumerate(zip(self.X, self.Y, self.fmts)):
            self.axes[count//2].plot(x, y, fmt)
            self.axes[count//2].set_ylabel(self.ylabel[count//2])
            self.axes[count//2].set_xlim(self.xlim)
            if count % 2 == 1:
                self.axes[count//2].legend(self.legend[count//2])

        plt.pause(0.1)

    def save_fig(self, generator_name, leadnum):
        """
        generator_name: The name of different generators.
        leadnum: 1-12
        """
        file_path = os.path.join(self.path, generator_name) # Load save file path
        filename = 'lead' + str(leadnum)
        file_path = os.path.join(file_path, filename)
        if not os.path.exists(file_path):
            os.makedirs(file_path) # Recursively create folder
        
        num = len(os.listdir(file_path))
        index = '_' + str(num+1)
        plt.savefig(os.path.join(file_path, 'ACGAN_pic'+index+'.png'))
        plt.show()