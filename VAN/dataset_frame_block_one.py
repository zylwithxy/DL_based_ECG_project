import os
from torch.utils.data import Dataset
import pandas as pd
import scipy.io as sio
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

class TraindataSet(Dataset):
    def __init__(self, TrainX, TrainY):
        super(TraindataSet, self).__init__()
        self.train = TrainX
        self.label = TrainY
    
    def __getitem__(self,index):
        return self.train[index], self.label[index]

    def __len__(self):
        return len(self.train)

class Read_data():
    def __init__(self, path, frame_index):
        """
        frame_index: [0, 9] int. choose the frame index.
        """
        self.F_l = 5500 # frame length 5500 length 
        self.F_n = 10 # the number of frames
        self.F_s = None # the frameshift
        self.index = frame_index
        
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

        return y_mlabel # (6877, 3)

    def read_data(self):

        X = []
        for item in self.result :
            item_path = os.path.join(self.path, item)
            sig = sio.loadmat(item_path)['ECG'][0][0][2]
            sos = signal.butter(8, 35, btype='low', output= 'sos', fs= 500) # Butterworth filtering
            filtered = signal.sosfilt(sos, sig) # shape (12, L)

            if filtered.shape[1] >= self.F_l + self.F_n - 1:
                character = self.seg_ecg_sig(filtered)
            else:
                character = self.ecg_sig_extract(filtered)
            X.append(character)
            
        return X
    
    def seg_ecg_sig(self, filtered):
        """
        filtered: shape(12, L). The filtered signal by butterworth
        """
        character = np.zeros((12, self.F_l), dtype = np.float32) # character: modified feature map

        S_l = filtered.shape[1] # The length of ecg signal
        self.F_s = (S_l - self.F_l) // (self.F_n - 1) # shift of segment signal
        start_cods = np.arange(0, S_l - self.F_l + 1, self.F_s) # ndarray Start coordinate

        try: 
            assert len(start_cods) >= 10 # The number frames
            cod = start_cods[self.index]
            character[:, :] = filtered[:, cod:cod+self.F_l]
        except Exception:
            print(f'The length of signal: {S_l}')
            print(f'The frame shift is {self.F_s}')
            print(f'The length of start_cods: {len(start_cods)}')
    
        return character # (12, self.F_l)

    def ecg_sig_extract(self, filtered):
        """
        This function is used for extracting signal with shorter length.
        """
        character = np.zeros((12, self.F_l), dtype = np.float32)
        length = filtered[:, :self.F_l].shape[1]
        character[:, :length] = filtered[:, :self.F_l]

        return character

# Plot the segment feature
def plot_seg_ecg(seg_ecg, index):
        """
        seg_ecg: segment ecg_signal
        index: A sample from 6877 samples
        """

        fig, axes = plt.subplots(12, sharex=True, 
                                 figsize=(15, 200),
                                    ) # 36
        axes = axes.ravel()

        seg_ecg = seg_ecg[index] # Choose the index

        
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

def label_count(label_first):

    label_count = []
    for i in np.arange(1, 10):
        label_count.append(np.array([label_first == i.astype(label_first.dtype)]).sum())
    return label_count

 
class Label_Indices():

    def __init__(self, label_first, dataset):
        """
        label_first: the first label of ECG samples.
        dataset: ndarray. 6877 training samples of ECG signals.
        """

        self.label_first = label_first
        self.label9_indices = self.label_indices_count()
        self.final9_indices = self.choose_indices()
        dataset = np.array(dataset)
        self.dataset = dataset[self.final9_indices]
        self.datalabel = self.label_first[self.final9_indices]

    def label_indices_count(self):

        label9_indices = [0] * 9

        for i in np.arange(1, 10):
            label9_indices[int(i-1)] = np.argwhere(self.label_first == i.astype(self.label_first.dtype)).reshape(-1)

        return label9_indices

    def choose_indices(self):

        indices = []
        for indice in self.label9_indices:   
            indices.append(indice[:150]) # 150: The number of STE samples included in training dataset
        
        return np.array(indices).reshape(-1)

if __name__ == "__main__":
    
    path = '/home/alien/XUEYu/paper_code/enviroment1'
    print('Start running')
    new_feature = Read_data(path, 4)
    # plot_seg_ecg(new_feature.train, 5)

    label_first = new_feature.labels[:, 0].reshape(-1) # 2D shape
    print(label_first.shape)
    print(type(label_first))
    fold_size = label_first.shape[0] // 4

    # idx = slice(fold_size, label_first.shape[0])
    idx = slice(0, fold_size)

    # label_count = label_count(label_first[idx]) # shape (9, indices_len)
    # print(label_count)

    label_count_all = label_count(label_first)
    print(label_count_all)

    """
    label_indices = Label_Indices(label_first[idx], new_feature.train)
    print(label_indices.dataset.shape)
    print(label_indices.datalabel.shape)
    """