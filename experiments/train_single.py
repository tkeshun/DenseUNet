import os
import sys 
from datetime import datetime
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from modules.config import setting_parse 
from modules.config import get_config 
from modules.loaders import SingleDataloader
from modules.loaders import load_model
from modules.option import EarlyStopping


def main() -> None:
    dtime = datetime.now()
    #引数のparse
    args = setting_parse()
    #実験の設定
    ##configuration
    config = get_config(args.config)
    do_time = "{0:%Y%m%d_%H}".format(dtime)#Colabで実行すると時間ずれる
    result_path = os.path.dirname(args.config)
 
    save_model_dir = os.path.join(result_path,"model{}".format(do_time))#learned model
    log_dir = os.path.join(result_path,"loss{}".format(do_time))
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(save_model_dir,exist_ok=True)
 
    valid_log = os.path.join(log_dir,"valid")
    train_log = os.path.join(log_dir,"train")
    os.makedirs(valid_log,exist_ok=True)
    os.makedirs(train_log,exist_ok=True)
    filename = os.path.join(save_model_dir,"model{}".format(do_time))
    pliod_filename = filename + "epoch"
 
     ##dataloaderの設定
    tr_loader, vl_loader ,tr_steps_per_epoch, vl_steps_per_epoch = SingleDataloader(path=config.train_data,
                                                                             inst=args.inst,
                                                                             batch_size=config.batch_size,
                                                                             rate=config.rate)
 
    ##モデルのロード
    model ,dev = load_model(config.model)
 
    ##optimizerの設定
    optimizer =  optim.Adam(model.parameters(),
                         lr=config.learning_rate,
                         betas=(0.9, 0.999), 
                         eps=1e-08, 
                         weight_decay=0, 
                         amsgrad=False) 
    ##loss
    criterion = nn.L1Loss()

    #FP16
    scaler = torch.cuda.amp.GradScaler() 
    

    ##training過程の設定
    def train_step(train_X, train_y):
        # 訓練モードに設定
        model.train()


        # フォワードプロパゲーションで出力結果を取得

        optimizer.zero_grad()   # 勾配を0で初期化（※累積してしまうため要注意）
        with torch.cuda.amp.autocast():
            pred_y = model(train_X) # 出力結果
            loss = criterion(pred_y, train_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得、数個前のセルで宣言    
        scaler.scale(loss).backward() # スケールした勾配を作る
        scaler.step(optimizer)# 勾配をアンスケールしてパラメータの更新
        # スケーラーの更新
        scaler.update()  
        item = loss.item()
        #del loss1,loss2,loss3,loss4
        #optimizer.step()  # 指定されたデータ分の最適化を実施
       
        return item# ※item()=Pythonの数値
 
    def valid_step(valid_X, valid_y):
        # 評価モードに設定（※dropoutなどの挙動が評価用になる）
        model.eval()
        # フォワードプロパゲーションで出力結果を取得
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_y = model(valid_X) # 出力結果
                loss_v = criterion(pred_y, valid_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得、数個前のセルで宣言
        item = loss_v.item()

        return item
  
 
    # 変数（学習／評価時に必要となるもの）
    avg_loss = 0.0           # 「訓練」用の平均「損失値」
    avg_val_loss = 0.0       # 「評価」用の平均「損失値」
    count = 0
    best_val_loss = np.inf
 
    # 損失の履歴を保存するための変数
    train_history = []
    valid_history = []

 
 
 
    for epoch in range(config.max_epochs):
        # forループ内で使う変数と、エポックごとの値リセット
        total_loss = 0.0     # 「訓練」時における累計「損失値」
        total_val_loss = 0.0 # 「評価」時における累計「損失値」
        total_train = 0      # 「訓練」時における累計「データ数」
        total_valid = 0      # 「評価」時における累計「データ数」
        save_flag = False
        print("Mixture to {}".format(args.inst))
        print('train_step')

        for Mix,inst in tr_loader:
            # 【重要】1ミニバッチ分の「訓練」を実行
            
            Mix,inst = Mix.to(dev),inst.to(dev) 
            loss = train_step(Mix,inst)

            # 取得した損失値と正解率を累計値側に足していく
            total_loss += loss          # 訓練用の累計損失値
            total_train += len(Mix) # 訓練データの累計数
        print('validation_step')   
        #学習
        for Mix_v,inst_v  in vl_loader:
            
            Mix_v,inst_v = Mix_v.to(dev),inst_v.to(dev) 
            # 【重要】1ミニバッチ分の「評価（精度検証）」を実行
            with torch.no_grad():
                val_loss = valid_step(Mix_v, inst_v)

            # 取得した損失値と正解率を累計値側に足していく
            total_val_loss += val_loss  # 評価用の累計損失値
            total_valid += len(Mix_v) # 訓練データの累計数
        n = epoch+1
        # ミニバッチ単位で累計してきた損失値や正解率の平均を取る
        
        avg_loss = total_loss / tr_steps_per_epoch                # 訓練用の平均損失値

        avg_val_loss = total_val_loss / vl_steps_per_epoch         # 評価用の平均損失値
        
        best_val_loss ,count,save_flag= EarlyStopping(avg_val_loss,best_val_loss,count)
        if save_flag:
            F_name = filename + ".pth"
            torch.save(model.state_dict(), F_name)
            #torch.save(optimizer.state_dict(),save_opt)
            print('save model {}'.format(F_name))
        if n%10 == 0 and n != 1:
            p_filename = pliod_filename + (str(n)+".pth") 
            torch.save(model.state_dict(),p_filename)
            #save_opt = os.path.join(save_model_dir,'optim_epoch{}.pth'.format(n))
            #torch.save(optimizer.state_dict(),save_opt)
        # グラフ描画のために損失の履歴を保存する
        train_history.append(avg_loss)
        valid_history.append(avg_val_loss)

        if n%10 == 0:
            print("save log")
            np.save(train_log,train_history)
            np.save(valid_log,valid_history)

        print("Epoch {:3d}/{:3d} loss:{:.10f}## val_loss:{:.7f} ".format(epoch+1, config.max_epochs, avg_loss,  avg_val_loss))

        if count > config.patience:
            np.save(train_log,train_history)
            np.save(valid_log,valid_history)
            print("Exceeding the patience , Early stopping!")
            break



    print("save log")
    np.save(train_log,train_history)
    np.save(valid_log,valid_history)

    print('Finished Training')


if __name__ == '__main__':
    main()
