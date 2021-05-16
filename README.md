# DenseUNet

UNet for MSS

## モデルの管理

今のところはパラメータが違ったら別モデルとして作成する
modelはclass model(): とし定義してその中に書く

## 実験の管理

### 実験設定の管理

argparse , yaml , dataclass

- datasets

  生データを置く
  datasets/データ名/ダウンロードした本体
  前処理が必要な場合は一緒に前処理プログラムを置く

- experiments
   実験用設定ファイル、実行ファイル、実験結果を置く
  
- moudules
  
  実験で使う機能(データローダ,model,性能試験, 前処理したデータ ...)  
  

```
.
├── README.md
├── createDir.sh
├── datasets
├── experiments
├── moudules
│   └── loaders
│       └── model_op.py
├── requirements.txt
├── scripts
├── setup.cfg
├── survey
├── tests
└── utils

``` 

