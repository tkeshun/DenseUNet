import numpy as np
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torch 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#データローダー関連
def dataloader(path,batch_size,rate):
    with h5py.File(path,'r') as f:
        mixture = torch.from_numpy(f['mixture'][:,0:512,:]).float()
        bass    = torch.from_numpy(f['bass'][:,0:512,:]).float()
        drums   = torch.from_numpy(f['drums'][:,0:512,:]).float()
        vocals  = torch.from_numpy(f['vocals'][:,0:512,:]).float()
        other   = torch.from_numpy(f['other'][:,0:512,:]).float()
    
    #datasetのランダム分割
    seed = 42
    torch.manual_seed(seed)
    val_size = int(mixture.shape[0]*rate)
    train_size = mixture.shape[0] - val_size
    mixture = mixture[:,np.newaxis]
    bass    = bass[:,np.newaxis]
    drums   = drums[:,np.newaxis]
    vocals  = vocals[:,np.newaxis]
    other   = other[:,np.newaxis]

    tr_steps_per_epochs = float((train_size + batch_size - 1)// batch_size)
    vl_steps_per_epochs = float((val_size + batch_size - 1)// batch_size)

    dataset_train = TensorDataset(mixture[:train_size],bass[:train_size],drums[:train_size],vocals[:train_size],other[:train_size])
    dataset_valid = TensorDataset(mixture[train_size:],bass[train_size:],drums[train_size:],vocals[train_size:],other[train_size:])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=2, pin_memory=True)
    torch.manual_seed(torch.initial_seed())
    return loader_train,loader_valid, tr_steps_per_epochs, vl_steps_per_epochs

def SingleDataloader(path,batch_size,inst="vocals",rate=0.8):
    with h5py.File(path,'r') as f:
        mixture = torch.from_numpy(f['mixture'][:,0:512,:]).float()
        inst    = torch.from_numpy(f[inst][:,0:512,:]).float()
    
    #datasetのランダム分割
    seed = 42
    torch.manual_seed(seed)
    val_size = int(mixture.shape[0]*rate)
    train_size = mixture.shape[0] - val_size
    mixture = mixture[:,np.newaxis]
    inst    = inst[:,np.newaxis]

    tr_steps_per_epochs = float((train_size + batch_size - 1)// batch_size)
    vl_steps_per_epochs = float((val_size + batch_size - 1)// batch_size)

    dataset_train = TensorDataset(mixture[:train_size],inst[:train_size])
    dataset_valid = TensorDataset(mixture[train_size:],inst[train_size:])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size)
    torch.manual_seed(torch.initial_seed())
    print("split data {:.2f}:{:.2f}".format(rate,1-rate))
    return loader_train,loader_valid, tr_steps_per_epochs, vl_steps_per_epochs


#モデル操作関連

"""
実装予定機能
指定したパスでのmodelのロード
モデルの保存(名付けなど自動化)
"""

from importlib import import_module

#モデルパスで指定したモデルのimport, load

def load_model(model_path):
  
    #model_pathで指定したモデルをimport
    NET = import_module(model_path) 
    #インスタンス化
    arc = NET.model()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arc = arc.to(dev)
    return arc, dev
