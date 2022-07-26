import os
from torch.utils.data import Dataset
import pandas as pd
import scipy.io as sio
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

class TraindataSet(Dataset):
    def __init__(self,TrainX, TrainY):
        super(TraindataSet, self).__init__()
        self.train = TrainX
        self.label = TrainY
    
    def __getitem__(self,index):
        return self.train[index], self.label[index]

    def __len__(self):
        return len(self.train)

# 想实现的思路是不要把所有的数据一口气全部读进来，用什么读什么，但这样花费时间更长，还是一气儿全部读进来吧
# 弄一个读取数据的大类
class Read_data():
    def __init__(self, path):

        self.F_l = 2000 # frame length
        self.F_n = 10 # the number of frames
        self.F_s = None # the frameshift
        
        self.path = path
        self.result = self.sort_data()
        self.labels = self.read_ylabel()
        self.train = self.read_data()
 
    def output(self):
        return (self.train, self.labels)
    
    def sort_data(self):
        result = []
        for mat_item in os.listdir(self.path):
            if mat_item.endswith('.mat'):
                result.append(mat_item)
        result.sort()

        assert len(result) == 6877
        return result
    
    def read_ylabel(self):
        '''
        Output:
        y_single_label: The first label of the training data
        y_mlabel: Multiple labels of the training data

        '''
        dataframe = pd.read_csv(os.path.join(self.path, 'REFERENCE1.csv'))
        y_mlabel = dataframe[['First_label','Second_label','Third_label']].values
        # _single_label = dataframe['First_label'].values
        # y_single_label = y_single_label.reshape(-1,1)
        assert len(y_mlabel) == 6877

        return y_mlabel

    def read_data(self):

        X = []
        for item in self.result :
            item_path = os.path.join(self.path, item)
            sig = sio.loadmat(item_path)['ECG'][0][0][2]
            sos = signal.butter(8, 35, btype='low', output= 'sos', fs= 500) # Butterworth filtering
            filtered = signal.sosfilt(sos, sig) # shape (12, L)
            character = self.seg_ecg_sig(filtered)
            X.append(character)
            
        return X
    
    def seg_ecg_sig(self, filtered):
        """
        filtered: shape(12, L). The filtered signal by butterworth
        """
        character = np.zeros((10, 12, 2000), dtype = np.float32) # character: modified feature map

        S_l = filtered.shape[1] # The length of ecg signal
        self.F_s = (S_l - self.F_l) // (self.F_n - 1) # shift of segment signal
        start_cods = np.arange(0, S_l - self.F_l + 1, self.F_s) # ndarray Start coordinate

        try: 
            assert len(start_cods) == 10 # The number frames
            for index, cod in enumerate(start_cods):
                character[index, :, :filtered.shape[1]] = filtered[:, cod:cod+self.F_l]
        except Exception:
            print(f'The length of signal: {S_l}')
            print(f'The frame shift is {self.F_s}')
            print(f'The length of start_cods: {len(start_cods)}')

        
        return character

# Plot the segment feature
def plot_seg_ecg(seg_ecg, index, frag_index):
        """
        seg_ecg: segment ecg_signal
        index: A sample from 6877 samples
        frag_index: 0 - 9
        """

        fig, axes = plt.subplots(12, sharex=True, 
                                 figsize=(15, 200),
                                    ) # 36
        axes = axes.ravel()

        seg_ecg = seg_ecg[index] # Choose the index
        seg_ecg = seg_ecg[frag_index] # Choose the fragment index

        
        for i in np.arange(12): # lead
            axes[i].grid(which='both', axis='both', linestyle='--')
            axes[i].plot(seg_ecg[i], color="salmon", label= 'ECG signal')
            axes[i].set_title(f'Lead {i+1}')
            if i == 0:
                axes[i].legend(loc= 'upper right')
        
        fig.tight_layout()
        top, bottom = fig.subplotpars.top, fig.subplotpars.bottom
        fig.subplots_adjust(top= top - 0.025, bottom= bottom + 0.024, hspace=0.8)
        print(fig.subplotpars.top, fig.subplotpars.bottom)
        plt.show()

if __name__ == "__main__":
    
    path = '/home/alien/XUEYu/paper_code/enviroment1'
    print('Start running')
    new_feature = Read_data(path)
    plot_seg_ecg(new_feature.train, 5, 6)
    """
    train_array = np.array(new_feature.train)
    print(train_array.shape)
    print(train_array[5][9][10][1000])
    pass
    """