import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',required=True)
    args = parser.parse_args()
    with open(args.file_path,"r") as f:
        data = json.load(f)
    
    total = []
    for name in data.keys():
        if name == "mean":
            continue
        total.append(data[name])

    dic = sorted(total,reverse=True)
    
    print("Mean   SDR : {:.5f}".format(np.mean(total)))
    print("Median SDR : {:.5f}".format(np.median(total)))
    for value in dic:
        print(value)

if __name__ == '__main__':
    main()
    
    