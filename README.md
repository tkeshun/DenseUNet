# 実験環境構築お試し

コードが散乱した実験環境から整理された実験環境を目指し, インターネット上で探したtipsを導入，試していく

##モデルの管理

今のところはパラメータが違ったら別モデルとして作成する
modelはclass model(): とし定義してその中に書く

##実験の管理

###実験設定の管理

argparse x yaml x dataclass


##log, 実験結果の管理
<ID>_<試行回数>_<yamlファイル名>_<実験内容がわかる名前>

##拡張アイディア

- [x] モデルを指定したときに動的なimportをしたい
 
importlib

##参考にしたサイト

https://blog.cormoran-web.com/blog/2019/12/ml-reserch-for-iq1/

https://www.slideshare.net/cvpaperchallenge/cvpaperchallenge-tips-241914101

##ディレクトリの説明

構造はtreeコマンドで描画

- datasets

  生データを置く
  datasets/データ名/ダウンロードした本体
  前処理が必要な場合は一緒に前処理プログラムを置く

- experiments
  
  実験スクリプト
  
   - <実験名>/
       README.md, 実験設定ファイル, 実行ファイルを置く
  

- moudules
  
  実験で使う機能(データローダ,model,性能試験, 前処理したデータ ...)  

- scripts

- survey
  論文pdf, 調査メモなど置く
- tests
  テストコード
- utils
  
  上のディレクトリ分類にないファイルを置く

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
# DenseUNet
