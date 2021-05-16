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

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mag(file,sr=16000,n_fft=1024,win_l=1024,hop_l=256,split_size=128):
    print("load_mag")

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
    print("complete load_mag")
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
    parser.add_argument('--savedir',required=True)
    parser.add_argument('--data_path',required=True)
    parser.add_argument('--model_path',required=True)
    parser.add_argument('--model_type',required=True)
    parser.add_argument('--sr',default=16000,type=int)
    parser.add_argument('--n_fft',default=1024,type=int)
    parser.add_argument('--win_l',default=1024,type=int)
    parser.add_argument('--hop_l',default=256,type=int)
    parser.add_argument('--list_path',required=True)
    parser.add_argument('--tag',required=True,default="test")
    parser.add_argument('--limit',type=int,default=50)
    parser.add_argument('--inst',required=True)
    args = parser.parse_args()



    save_dir = args.savedir
    data_path  = args.data_path
    model_path = args.model_path
    tag = args.tag
    #param
    split_size = args.input_size
    sr = args.sr
    n_fft = args.n_fft
    win_l = args.win_l
    hop_l = args.hop_l
    list_path = args.list_path
    limit = args.limit

    audio_list = load_list(list_path)
    ##モデルのロード
    model ,dev = load_model(args.model_type)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    cnt = 1
    for path in audio_list:
        if limit == 0:
            print("limit. finish!")
            break
        print(path)
        cnt+=1
        orig_path = os.path.join(data_path,tag,path)
        orig_path = os.path.join(orig_path,"mixture.wav")
        split_mag,split_phase ,max_value = load_mag(orig_path)
        sep_song = compute_separation(split_mag=split_mag,
                                    split_phase=split_phase,
                                    split_size=split_size,
                                    hop_l=hop_l,
                                    win_l=win_l,
                                    max_v=max_value,
                                    model=model)
        save_path = os.path.join(save_dir,tag,path)
        os.makedirs(save_path, exist_ok=True)
        inst_name = args.inst+'.wav'
        sf.write(os.path.join(save_path,inst_name), sep_song, sr)
        
        limit-=1

    print("save file,Complete all programs.")


if __name__ == "__main__":
    main()