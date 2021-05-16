#!/bin/bash
#experimentディレクトリで実行する
SAVEFILE="./unet_vocal_single/result/Experiment/"
DATA_PATH="./dataset/orig_mono16000"
tag="test"
MODEL_TYPE="modules.models.unet_single"
MODEL_PATH="./unet_vocal_single/result/model20210405_14/model20210405_14.pth"
list_path="./list.txt"
limit=50
python3 test_single.py --input_size 128 --savedir $SAVEFILE --data_path  $DATA_PATH --model_path $MODEL_PATH --sr 16000 --n_fft 1024 --win_l 1024 --hop_l 256 --list_path $list_path --tag $tag --limit $limit --model_type $MODEL_TYPE --inst "vocals"
#chmod 777 ./unet_vocal_single/result/Experiment/