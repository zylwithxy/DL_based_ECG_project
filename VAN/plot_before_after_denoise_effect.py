import os
from torch.utils.data import Dataset
import pandas as pd
import scipy.io as sio
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Show the effect of butterworth filter
class Butter_effect():
    def __init__(self, path, index= 0) -> None:
        self.record_path = path # File location
        self.result = self.sort_data()
        self.orig, self.filtered = self.read_data(index) 
    
    def sort_data(self):
        result = []
        for mat_item in os.listdir(self.record_path):
            if mat_item.endswith('.mat'):
                result.append(mat_item)
        
        result.sort()

        assert len(result) == 6877
        return result
    
    def read_data(self, index):
        """
        index : The index of samples
        """
        sample_name = self.result[index]
        
        item_path = os.path.join(self.record_path, sample_name)
        sig = sio.loadmat(item_path)['ECG'][0][0][2]
        sos = signal.butter(8, 35, btype='low', output= 'sos', fs= 500)
        filtered = signal.sosfilt(sos, sig)
        return sig, filtered
            
    def plot_ecg(self, orig, filtered):
        """
        orig: original signal: shape(12, L)
        filtered: filtered signal: shape(12, L)
        """

        fig, axes = plt.subplots(12, sharex=True, figsize=(15, 36))
        axes = axes.ravel()
        
        for i in np.arange(12):
            axes[i].grid(which='both', axis='both', linestyle='--')
            axes[i].plot(orig[i], color="salmon", label= 'Original')
            axes[i].plot(filtered[i], color="b", label = 'After butter')
            axes[i].set_title(f'Lead {i+1}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    path = '/home/alien/XUEYu/paper_code/enviroment1'
    print(path)
    butter_effect = Butter_effect(path, 2)
    butter_effect.plot_ecg(butter_effect.orig, butter_effect.filtered)
    # print(butter_effect.orig.shape, butter_effect.filtered.shape)