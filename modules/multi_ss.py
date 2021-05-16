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
    
    optimizer.step()  # 指定されたデータ分の最適化を実施
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
