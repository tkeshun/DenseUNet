#!/bin/bash
#experimentディレクトリで実行する
SAVEFILE="offset_hist_df8_in_mixture_to_vocals_probability"
FILEPATH="../experiments/dataset/orig_mono16000/test/Al James - Schoolboy Facination/vocals.wav"
MODEL_TYPE="modules.models.DenseUnet_b8_deform"
LIST_PATH="../experiments/dataset/orig_mono16000/test"
#MODEL_PATH="../experiments/DenseUnet_vocals/result/model20210420_11/model20210420_11.pth"
MODEL_PATH="../experiments/DenseUnet_vocals_deform_b8/result/model20210423_21/model20210423_21.pth"
python3 vision.py --input_size 128 --savefile $SAVEFILE  --model_path $MODEL_PATH --list_path $LIST_PATH --model_type $MODEL_TYPE 
#chmod 666 ./DenseUnet_vocals/result/Experiment/