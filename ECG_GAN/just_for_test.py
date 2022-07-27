import pandas as pd
import os
import torch
from Generator_model import Generator_3 as Generator

CLASSES = 9
HIDDEN_DIM = 400
L = 5500

generator = Generator(CLASSES, HIDDEN_DIM, L)
print(generator.__class__)
print(generator.__class__.__name__)
print(type(generator.__class__.__name__))

PATH = '/home/alien/XUEYu/paper_code/enviroment1'
dataframe = pd.read_csv(os.path.join(PATH, 'REFERENCE1.csv'))
y_mlabel = dataframe[['First_label','Second_label','Third_label']].values

print(y_mlabel.shape)
print(y_mlabel[0])

print(torch.tensor(y_mlabel[0], dtype= torch.int64)) # For nan: nan -> -9223372036854775808