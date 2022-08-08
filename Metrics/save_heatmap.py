import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_heatmap(mat_group, label):
    for i in range(4):
        temp = pd.DataFrame(mat_group[i], index = label, columns = label)
        ax = sns.heatmap(temp, cmap='Greens', annot=True, fmt="d")
        # ax = sns.heatmap(temp, cmap='Blues', annot=True, fmt="d")
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        # print(mat_group[i].sum())
        plt.show()
        # plt.savefig()

def plot_single_heatmap(single_mat, label):

    single_mat = single_mat[0]
    temp = pd.DataFrame(single_mat, index = label, columns = label)
    ax = sns.heatmap(temp, cmap='Greens', annot=True, fmt="d")
    # ax = sns.heatmap(temp, cmap='Blues', annot=True, fmt="d")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    # print(mat_group[i].sum())
    plt.show()


def cal_F1(confuse_matrix):

    F11 = 2 * confuse_matrix[0][0] / (np.sum(confuse_matrix[0, :]) + np.sum(confuse_matrix[:, 0]))
    F12 = 2 * confuse_matrix[1][1] / (np.sum(confuse_matrix[1, :]) + np.sum(confuse_matrix[:, 1]))
    F13 = 2 * confuse_matrix[2][2] / (np.sum(confuse_matrix[2, :]) + np.sum(confuse_matrix[:, 2]))
    F14 = 2 * confuse_matrix[3][3] / (np.sum(confuse_matrix[3, :]) + np.sum(confuse_matrix[:, 3]))
    F15 = 2 * confuse_matrix[4][4] / (np.sum(confuse_matrix[4, :]) + np.sum(confuse_matrix[:, 4]))
    F16 = 2 * confuse_matrix[5][5] / (np.sum(confuse_matrix[5, :]) + np.sum(confuse_matrix[:, 5]))
    F17 = 2 * confuse_matrix[6][6] / (np.sum(confuse_matrix[6, :]) + np.sum(confuse_matrix[:, 6]))
    F18 = 2 * confuse_matrix[7][7] / (np.sum(confuse_matrix[7, :]) + np.sum(confuse_matrix[:, 7]))
    F19 = 2 * confuse_matrix[8][8] / (np.sum(confuse_matrix[8, :]) + np.sum(confuse_matrix[:, 8]))

    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19) / 9

    return float(F1), (F11, F12, F13, F14, F15, F16, F17, F18, F19), [np.sum(confuse_matrix[i, :]) for i in range(9)]

if __name__ == '__main__':

    # PATH = '/home/alien/XUEYu/paper_code/Parameters/2018_torch' # file storage road
    # PATH = '/home/alien/XUEYu/paper_code/Parameters/2018_Multi'
    PATH = ['/home/alien/XUEYu/paper_code/Parameters/2018_torch',
            '/home/alien/XUEYu/paper_code/Parameters/2018_Resnet',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_600',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_2/VAN_3_L_5500_fold1',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_2/VAN_3_L_5500_fold1_version_1',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE&PAC_600',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE&PAC_600/Experiment2',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE&PAC_600/Experiment3',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE&PAC_600/Experiment4',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE&PAC_600_900',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE_150',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE_300',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE_600',
            '/home/alien/XUEYu/paper_code/Parameters/2018_torch/Data_aug_5500',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_1',
            '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_2',
            ]

    # mat_group = [np.load(os.path.join(PATH[1], '1fold Test_model_confuse_mat.npy')).astype(np.int32) for i in range(1, 5)]
    # mat_group = [np.load(os.path.join(PATH[2], str(1)+'fold Test_model__confuse_mat.npy')).astype(np.int32)]
    # mat_group = [np.load(os.path.join(PATH[-4], '1fold Test_model_confuse_mat.npy')).astype(np.int32)]
    mat_group = [np.load(os.path.join(PATH[-4], '1fold Test_model_confuse_mat.npy')).astype(np.int32)]
    
    """
    mat_group = [np.load(os.path.join(PATH[-3], \
                'Label8_Sample150_1fold Test_model_confuse_mat5.npy')).astype(np.int32)]
    """
   
    label = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    # Calculate results
    max_result, branch_result, type_result = [], [], []
    for item in mat_group:
        temp1, temp2, temp3 = cal_F1(item)
        max_result.append(temp1)
        branch_result.append(temp2)
        type_result.append(temp3)

    print(f'Four fold cross validation best results are seperately: {max_result}\n')
    
    for num, branch in enumerate(branch_result):
        print(f"{num+1} fold test result{branch}\n")

    for num, type in enumerate(type_result):
        print(f"{num+1} fold test type{type}\n")

    plot_single_heatmap(mat_group, label)
    # plot_heatmap(mat_group, label)