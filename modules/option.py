import numpy as np
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torch

def EarlyStopping(loss,best_loss,count):
    flag =False
    if loss > best_loss:
        print('The validation loss did not improve from {}.'.format(best_loss))
        count += 1
    else:
        print('The validation loss has been improved from {} to {}'.format(best_loss,loss))
        best_loss = loss
        count = 0
        flag = True
    return best_loss,count,flag
 
