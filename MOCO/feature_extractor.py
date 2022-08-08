import sys

# Using STA_CRNN as an anchor
PATH = '/home/alien/XUEYu/paper_code/enviroment1'
PATH2 = '/home/alien/XUEYu/paper_code/enviroment1/CPC'

sys.path.append(PATH)
sys.path.append(PATH2)

from STA_CRNN.cpsc2018_torch import vgg, Read_data
from dataset_CPC import data_generate, TraindataSet_total 