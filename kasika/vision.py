import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
from datetime import datetime
import argparse
import pickle
from modules.loaders import load_model
import librosa
from librosa import stft,istft
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mag(file,sr=16000,n_fft=1024,win_l=1024,hop_l=256,split_size=128):
    #print("load_mag")

    y, sr = librosa.load(file,sr=sr,mono=True)
    spec = librosa.stft(y,n_fft=n_fft,win_length=win_l,hop_length=hop_l)
    #振幅に変換
    mag = np.abs(spec)
    #最大値正規化
    max_value = np.max(mag)
    mag /= max_value
    phase = np.exp(1.j*np.angle(spec))
    #サイズをinput_sizeの倍数に合わせる
    if mag.shape[1] % split_size == 0:
        seq_number = int(mag.shape[1] / split_size)
    else:
        seq_number = int(mag.shape[1] / split_size) + 1
        pad_value = split_size - mag.shape[1] % split_size
        mag = np.pad(mag,((0,0),(0,pad_value)),mode='constant',constant_values=0.0)
        phase = np.pad(phase,((0,0),(0,pad_value)),mode='constant',constant_values=0.0)
    split_mag = []
    split_phase = []
    for frame in range(seq_number):
        tmp_mag = mag[:,frame*split_size : frame*split_size+split_size]
        tmp_phase = phase[:,frame*split_size : frame*split_size+split_size]
        split_mag.append(tmp_mag)
        split_phase.append(tmp_phase)
        #print('process number {0} / {1}'.format(frame,seq_number))
    split_mag = np.array(split_mag)
    split_phase = np.array(split_phase)  
    #print("complete load_mag")
    return split_mag,split_phase,max_value

def compute_audio(input_mag,model):
    input_mag = torch.from_numpy(input_mag).float()
    input_mag = input_mag.to(dev)
    audio = model(input_mag)
    audio = audio.to('cpu').detach().float().numpy()         
    return audio

def compute_separation(split_mag,split_phase,split_size,hop_l,win_l,max_v,model):

    all_mag = []
    for i in range(split_mag.shape[0]):
        input_spec = split_mag[i:i+1,0:512]
        input_spec = input_spec[:,np.newaxis]
        mag_sp = compute_audio(input_spec,model)#分離        
        
        all_mag.append(mag_sp)
    

    all_mag = np.array(all_mag)
    #楽器復元のための前処理
    
    sep_song = np.empty(0)
    index = 0
    for audio in all_mag: 
        audio = np.reshape(audio,[1,512,128])
        #mag   = np.reshape(split_mag[index,0:512],[512,128]) 
        phase = np.reshape(split_phase[index,0:512],[1,512,128])
        
        #bin数合わせに0^padding
        
        pad_bin = np.zeros([1,1,split_mag.shape[2]],dtype="float32")
        comp = np.concatenate([phase * audio,pad_bin],1)

        #断片をarrayに追加
        
        if index == 0:
            sep_song = comp
            index+=1
            continue
        
        index+=1
        sep_song = np.concatenate([sep_song,comp],2)
        
    print("shape:",sep_song.shape,end="")
    
    sep_song = np.reshape(sep_song,[len(sep_song[0]),len(sep_song[0][0])])
    sep_song*=max_v
    wav = istft(sep_song,hop_length=hop_l,win_length=win_l)

    print(" Completed istft, return narray()")
    
    return wav

def load_list(path):
    with open(path,'rb') as f:
        list_path = pickle.load(f)
    return list_path

def main():
    print('program start')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size',required=True,type=int)
    parser.add_argument('--savefile',required=True)
    parser.add_argument('--model_path',required=True)
    parser.add_argument('--model_type',required=True)
    parser.add_argument('--sr',default=16000,type=int)
    parser.add_argument('--n_fft',default=1024,type=int)
    parser.add_argument('--win_l',default=1024,type=int)
    parser.add_argument('--hop_l',default=256,type=int)
    parser.add_argument('--list_path',required=True)
    #parser.add_argument('--file_path,required=True)#可視化実験に使うファイル

    args = parser.parse_args()
    save_file = args.savefile

    model_path = args.model_path
    #param
    split_size = args.input_size
    sr = args.sr
    n_fft = args.n_fft
    win_l = args.win_l
    hop_l = args.hop_l
    #file_path = args.file_path
    
    dirs = os.listdir(args.list_path)


    #学習済みモデルのロード
    model ,dev = load_model(args.model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #print(model)
    #target_module = model.dense7_c3.p_conv
    target_module = model.dense8_c3.p_conv
    features = None
    def extract(target, inputs):
        global features
        features = None
        def forward_hook(module, inputs, outputs):
            # 順伝搬の出力を features というグローバル変数に記録する
            global features
            features = outputs.to('cpu').detach()

            #print("--features--",features)
            # コールバック関数を登録する。
        handle = target.register_forward_hook(forward_hook)
        model.eval()
        model(inputs)
        handle.remove()
        #print(features)
        return features
    
    tmp_x = []
    tmp_y = []
    for name in dirs:
        print(name)
        file_path = os.path.join(args.list_path,name,"mixture.wav")
        split_mag,split_phase ,max_value = load_mag(file_path)
        
        for i in range(len(split_mag)//2):
            input_mag = split_mag[i:i+1,0:512]
            input_mag = input_mag[:,np.newaxis]
            input_mag = torch.from_numpy(input_mag).float()
            input_mag = input_mag.to(dev)
            feature = extract(target_module, input_mag)
            feature = feature.float().numpy()
            F_x = abs(feature[:,:9])
            F_y = abs(feature[:,9:])
            tmp_x.append(copy.deepcopy(F_x))
            tmp_y.append(copy.deepcopy(F_y))
    

    save_x = save_file+"_x.svg"
    save_y = save_file+"_y.svg"


    tmp_x = np.ravel(np.array(tmp_x))
    
    print("Flatten")
    histplot_x = sns.histplot(tmp_x,binrange=[0,50],binwidth=0.1,stat="probability")
    fig_x = histplot_x.get_figure()
    fig_x.savefig(save_x,format="svg")
    print("save_x")
    del tmp_x
    plt.clf()
    tmp_y = np.ravel(np.array(tmp_y))    
    histplot_y = sns.histplot(tmp_y,binrange=[0,50],binwidth=0.1,stat="probability")
    fig_y = histplot_y.get_figure()
    fig_y.savefig(save_y,format="svg")
    print("save_y")
    #histplot = sns.histplot(FE,binrange=[-25,25],binwidth=1,stat="density")
    #histplot = sns.histplot(FE,binwidth=1,binrange=[-75,75])
    #histplot = sns.histplot(FE,binwidth=1,binrange=[-100,100])
    #plt.legend() # 凡例を表示
    

    

    

    #print("FILENAME: %s" % os.path.basename(args.file_path))
    

if __name__ == "__main__":
    main()