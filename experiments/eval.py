import argparse
import pickle
import os
from mir_eval.separation import bss_eval_sources
import librosa
import json
import numpy as np
import time

def load_list(path):
    with open(path,'rb') as f:
        list_path = pickle.load(f)
    return list_path

def main():
    print("start program")
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path',required=True)
    parser.add_argument('--estimates_dir',required=True)
    parser.add_argument('--subsets',default="Test")
    parser.add_argument('--list_path',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--savename',required=True)
    parser.add_argument('--inst',type=str,required=True)

    args = parser.parse_args()
    estimates_dir = args.estimates_dir
    output_dir    = args.output_dir
    subsets       = args.subsets
    list_path     = args.list_path
    original_path = args.original_path    
    sr = 16000
    audio_list = load_list(list_path)
    sdr_dict = {}
    inst = args.inst
    total = 0.0


    for name in audio_list:
        print("compute ",name)
        #曲のディレクトリ
        ref_dir = os.path.join(original_path,name)
        est_dir = os.path.join(estimates_dir,name)
        #結果格納変数
        
           
        
        #データロード    
        ref ,_ = librosa.load(os.path.join(ref_dir,inst+".wav"),sr=sr,mono=True)
        est ,_ = librosa.load(os.path.join(est_dir,inst+".wav"),sr=sr,mono=True)

        if len(ref) < len(est):
            len1 = len(ref) 
            len2 = len(est)
            pad = np.zeros(len2-len1)                
            ref = np.concatenate([ref,pad],0)

        librosa.util.normalize    
        reference = (librosa.util.normalize(ref))
        estimate  = (librosa.util.normalize(est))
        
        print("compute SDR")

        sdr ,_,_,_ = bss_eval_sources(reference,estimate)
        total+=sdr
        sdr_dict[name] = sdr[0]
        print(sdr_dict[name])
        #print("Finish!")
    #save sdr
    mean = float(total) / float(len(sdr_dict))
    sdr_dict["mean_sdr"] = mean
    print("Mean SDR : " ,mean)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,"sdr_{}.json".format(args.savename)),"w") as f:
        json.dump(sdr_dict,f,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    elapsed_time = t2-t1
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
