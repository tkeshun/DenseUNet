#!/bin/bash
yaml_path="./unet_vocal_single/result/exp.yaml"
python3 train_single.py --config $yaml_path --inst "vocals"
