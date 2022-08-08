"""
1. Solve attempted relative import with no known parent package
import sys
sys.path.insert(0, "..")
"""

"""
import os # 2
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, "./STA_CRNN")
from STA_CRNN.k_fold_CV import get_k_fold_data
from STA_CRNN.k_fold_CV import evaluate_accuracy_gpu
from STA_CRNN.CBAM import CBAMBlock
from STA_CRNN.cpsc2018_torch import try_gpu
from STA_CRNN.cpsc2018_torch import build_net
PATH = '/home/alien/XUEYu/paper_code/enviroment1/ECG_GAN'

"""

# print(os.getcwd())