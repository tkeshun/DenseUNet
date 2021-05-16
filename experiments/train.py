"""
作業進捗
未)
loadersの中身実装する
multi_ssの中身実装する
"""
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
from modules.loaders import dataloader
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
    os.makedirs(valid_log)
    os.makedirs(train_log)
    filename = os.path.join(save_model_dir,"model{}".format(do_time))
    pliod_filename = filename + "epoch"
 
     ##dataloaderの設定
    tr_loader, vl_loader ,tr_steps_per_epoch, vl_steps_per_epoch = dataloader(path=config.train_data,
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
     
    ##training過程の設定
    def train_step(train_X, train_y):
        # 訓練モードに設定
        model.train()
        # フォワードプロパゲーションで出力結果を取得
        pred_y = model(train_X) # 出力結果
        optimizer.zero_grad()   # 勾配を0で初期化（※累積してしまうため要注意）
        loss1 = criterion(pred_y[0], train_y[0])     # 誤差（出力結果と正解ラベルの差）から損失を取得、数個前のセルで宣言
        loss2 = criterion(pred_y[1], train_y[1])
        loss3 = criterion(pred_y[2], train_y[2])
        loss4 = criterion(pred_y[3], train_y[3])
        loss = (loss1+loss2+loss3+loss4)/4    
        loss.backward()
         
        item,item1, item2, item3, item4 = loss.item(), loss1.item(),loss2.item(),loss3.item(),loss4.item()
        #del loss1,loss2,loss3,loss4
        optimizer.step()  # 指定されたデータ分の最適化を実施
        del loss,loss1,loss2,loss3,loss4 
        gc.collect()
        return item, item1, item2, item3, item4 # ※item()=Pythonの数値
 
    def valid_step(valid_X, valid_y):
        # 評価モードに設定（※dropoutなどの挙動が評価用になる）
        model.eval()
        # フォワードプロパゲーションで出力結果を取得
        with torch.no_grad():
            pred_y = model(valid_X) # 出力結果
            loss1_v = criterion(pred_y[0], valid_y[0])     # 誤差（出力結果と正解ラベルの差）から損失を取得、数個前のセルで宣言
            loss2_v = criterion(pred_y[1], valid_y[1])
            loss3_v = criterion(pred_y[2], valid_y[2])
            loss4_v = criterion(pred_y[3], valid_y[3])
            loss_v = (loss1_v + loss2_v + loss3_v + loss4_v) / 4
        item ,item1, item2, item3, item4 = loss_v.item(),loss1_v.item(),loss2_v.item(),loss3_v.item(),loss4_v.item()
        del loss1_v,loss2_v,loss3_v,loss4_v
        gc.collect()
        return item,item1, item2, item3, item4
  
 
    # 変数（学習／評価時に必要となるもの）
    avg_loss = 0.0           # 「訓練」用の平均「損失値」
    avg_val_loss = 0.0       # 「評価」用の平均「損失値」
    count = 0
    best_val_loss = np.inf
 
    # 損失の履歴を保存するための変数
    train_history = []
    train_history1 = []
    train_history2 = []
    train_history3 = []
    train_history4 = []
    valid_history = []
    valid_history1 = []
    valid_history2 = []
    valid_history3 = []
    valid_history4 = []
 
 
 
 
    for epoch in range(config.max_epochs):
        # forループ内で使う変数と、エポックごとの値リセット
        total_loss = 0.0     # 「訓練」時における累計「損失値」
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss4 = 0.0
        total_val_loss = 0.0 # 「評価」時における累計「損失値」
        total_val_loss1 = 0.0
        total_val_loss2 = 0.0
        total_val_loss3 = 0.0
        total_val_loss4 = 0.0
        total_train = 0      # 「訓練」時における累計「データ数」
        total_valid = 0      # 「評価」時における累計「データ数」
        save_flag = False
        print('train_step')

        for Mix,Ba,Dr,Vo,Oth in tr_loader:
            # 【重要】1ミニバッチ分の「訓練」を実行
            
            Mix,Ba,Dr,Vo,Oth = Mix.to(dev),Ba.to(dev),Dr.to(dev),Vo.to(dev),Oth.to(dev) 
            loss,loss1,loss2,loss3,loss4 = train_step(Mix,[Ba,Dr,Vo,Oth])

            # 取得した損失値と正解率を累計値側に足していく
            total_loss += loss          # 訓練用の累計損失値
            total_loss1 += loss1
            total_loss2 += loss2
            total_loss3 += loss3
            total_loss4 += loss4
            total_train += len(Mix) # 訓練データの累計数
        print('validation_step')   
        #学習
        for Mix_v,Ba_v,Dr_v,Vo_v,Oth_v  in vl_loader:
            
            Mix_v,Ba_v,Dr_v,Vo_v,Oth_v = Mix_v.to(dev),Ba_v.to(dev),Dr_v.to(dev),Vo_v.to(dev),Oth_v.to(dev) 
            # 【重要】1ミニバッチ分の「評価（精度検証）」を実行
            with torch.no_grad():
                val_loss ,val_loss1,val_loss2,val_loss3,val_loss4= valid_step(Mix_v, [Ba_v,Dr_v,Vo_v,Oth_v])

            # 取得した損失値と正解率を累計値側に足していく
            total_val_loss += val_loss  # 評価用の累計損失値
            total_val_loss1 += val_loss1
            total_val_loss2 += val_loss2
            total_val_loss3 += val_loss3
            total_val_loss4 += val_loss4
            total_valid += len(Mix_v) # 訓練データの累計数
        n = epoch+1
        # ミニバッチ単位で累計してきた損失値や正解率の平均を取る
        
        avg_loss = total_loss / tr_steps_per_epoch                # 訓練用の平均損失値
        avg_loss1 = total_loss1 / tr_steps_per_epoch
        avg_loss2 = total_loss2 / tr_steps_per_epoch
        avg_loss3 = total_loss3 / tr_steps_per_epoch
        avg_loss4 = total_loss4 / tr_steps_per_epoch

        avg_val_loss = total_val_loss / vl_steps_per_epoch         # 評価用の平均損失値
        avg_val_loss1 = total_val_loss1 / vl_steps_per_epoch 
        avg_val_loss2 = total_val_loss2 / vl_steps_per_epoch 
        avg_val_loss3 = total_val_loss3 / vl_steps_per_epoch 
        avg_val_loss4 = total_val_loss4 / vl_steps_per_epoch 
        
        best_val_loss ,count,save_flag= EarlyStopping(avg_val_loss,best_val_loss,count)
        if save_flag:
            F_name = filename + ".pth"
            torch.save(model.state_dict(), F_name)
            #torch.save(optimizer.state_dict(),save_opt)
            print('save model {}'.format(filename))
        if n%10 == 0 and n != 1:
            p_filename = pliod_filename + (str(n)+".pth") 
            torch.save(model.state_dict(),p_filename)
            #save_opt = os.path.join(save_model_dir,'optim_epoch{}.pth'.format(n))
            #torch.save(optimizer.state_dict(),save_opt)
        # グラフ描画のために損失の履歴を保存する
        train_history.append(avg_loss)
        train_history1.append(avg_loss1)
        train_history2.append(avg_loss2)
        train_history3.append(avg_loss3)
        train_history4.append(avg_loss4)
        valid_history.append(avg_val_loss)
        valid_history1.append(avg_val_loss1)
        valid_history2.append(avg_val_loss2)
        valid_history3.append(avg_val_loss3)
        valid_history4.append(avg_val_loss4)

        if n%10 == 0:
            print("save log")
            loss_P1 = os.path.join(train_log,"train1")
            loss_P2 = os.path.join(train_log,"train2")
            loss_P3 = os.path.join(train_log,"train3")
            loss_P4 = os.path.join(train_log,"train4")
            np.save(train_log,train_history)
            np.save(loss_P1,train_history1)
            np.save(loss_P2,train_history2)
            np.save(loss_P3,train_history3)
            np.save(loss_P4,train_history4)
            loss_P1 = os.path.join(valid_log,"valid1")
            loss_P2 = os.path.join(valid_log,"valid2")
            loss_P3 = os.path.join(valid_log,"valid3")
            loss_P4 = os.path.join(valid_log,"valid4")
            np.save(valid_log,valid_history)
            np.save(loss_P1,valid_history1)
            np.save(loss_P2,valid_history2)
            np.save(loss_P3,valid_history3)
            np.save(loss_P4,valid_history4)

        print("Epoch {:3d}/{:3d} loss:{:.10f} ba_loss:{:.7f} dr_loss:{:.7f} vo_loss:{:.7f} oth_loss:{:.7f} ## val_loss:{:.7f} ba_vloss:{:.7f} dr_vloss:{:.7f} vo_vloss:{:.7f} oth_vloss:{:.7f}".format(epoch+1, config.max_epochs, avg_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4, avg_val_loss, avg_val_loss1, avg_val_loss2, avg_val_loss3, avg_val_loss4))

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
