#!/bin/bash
#experimentディレクトリで実行する
SAVEFILE=""
FILEPATH="../experiments/dataset/orig_mono16000/test/Al James - Schoolboy Facination/vocals.wav"
MODEL_TYPE="modules.models.DenseUnet_b7_deform"
#MODEL_PATH="../experiments/DenseUnet_vocals/result/model20210420_11/model20210420_11.pth"
MODEL_PATH="../experiments/DenseUnet_vocals_deform_b7/result/model20210423_16/model20210423_16.pth"
MODULE=""
python3 vision.py --input_size 128 --savefile $SAVEFILE --file_path  "../experiments/dataset/orig_mono16000/test/Al James - Schoolboy Facination/mixture.wav" --model_path $MODEL_PATH --sr 16000 --n_fft 1024 --win_l 1024 --hop_l 256 --model_type $MODEL_TYPE  
#chmod 666 ./DenseUnet_vocals/result/Experiment/