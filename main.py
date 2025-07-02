import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import shutil
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset import SynergyBothEncoderDataset
from torch.optim import AdamW
import os
from prettytable import PrettyTable
from model_h import BothNet
from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score#改成分类
from sklearn.metrics import balanced_accuracy_score,cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score,recall_score,precision_score
from utils import save_model, load_model, compute_kl_loss, AverageMeter



def get_data_loader_both(dataset_name, batch_size, split_way):
    train_loaders = []
    valid_loaders = []
    test_loaders = []
    for i in range(5):
        train_dataset = SynergyBothEncoderDataset(dataset_name=dataset_name, i=i, type='train',split=split_way)
        valid_dataset = SynergyBothEncoderDataset(dataset_name=dataset_name, i=i, type='val',split=split_way)
        test_dataset = SynergyBothEncoderDataset(dataset_name=dataset_name, i=i, type='test',split=split_way)
        train_loader = DataLoader( train_dataset,
                batch_size= batch_size, shuffle=True, num_workers=16)
        valid_loader = DataLoader( valid_dataset,
                batch_size= batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader( test_dataset,
                batch_size= batch_size, shuffle=False, num_workers=16)
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)
    return train_loaders, valid_loaders, test_loaders 


def validate_new(valid_loader, model):#valid_loader
    model.eval()#设为评估模式
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            (y, cell, drug1_2d_features, drug1_3d_features,drug2_2d_features, drug2_3d_features) = batch
            (y, cell, drug1_2d_features, drug1_3d_features,drug2_2d_features, drug2_3d_features) = (y.to(device), cell.to(device), 
                                                                    drug1_2d_features.to(device), drug1_3d_features.to(device),
                                                                    drug2_2d_features.to(device), drug2_3d_features.to(device))
            pre_synergy = model(cell, drug1_2d_features, drug1_3d_features, drug2_2d_features, drug2_3d_features)
            pre_synergy = torch.nn.functional.softmax(pre_synergy, dim=1)[:,1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)#cat:多个tensor拼接。删除pre_synergy.argmax(-1)，使其一维直接输出
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
    y_pred = np.array(preds) > 0.5#np.where(preds>0.5,1,0)
    accuracy = accuracy_score(trues, y_pred)
    BACC = balanced_accuracy_score(trues, y_pred)
    roc_auc = roc_auc_score(trues, preds)
    ACC = accuracy_score(trues, y_pred)
    F1 = f1_score(trues, y_pred, average='binary')
    Prec = precision_score(trues, y_pred, average='binary')
    Rec = recall_score(trues, y_pred, average='binary')
    kappa = cohen_kappa_score(trues, y_pred)
    mcc = matthews_corrcoef(trues, y_pred)
    ap = average_precision_score(trues, preds)
    return accuracy,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap


def train(train_loader, model, epoch, optimizer, device, scheduler):
    losses = AverageMeter()
    model.train()
    print_freq=200
    cross_entropy_loss = nn.CrossEntropyLoss()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        (y, cell, drug1_2d_features, drug1_3d_features,drug2_2d_features, drug2_3d_features) = batch
        (y, cell, drug1_2d_features, drug1_3d_features,drug2_2d_features, drug2_3d_features) = (y.to(device), cell.to(device), 
                                                                    drug1_2d_features.to(device), drug1_3d_features.to(device),
                                                                    drug2_2d_features.to(device), drug2_3d_features.to(device))
        pre_synergy = model(cell, drug1_2d_features, drug1_3d_features,drug2_2d_features, drug2_3d_features)
        pre_synergy2 = model(cell, drug1_2d_features, drug1_3d_features,drug2_2d_features, drug2_3d_features)
        ce_loss = 0.5 * (cross_entropy_loss(pre_synergy, y.squeeze(1)) + cross_entropy_loss(pre_synergy2, y.squeeze(1)))
        kl_loss = compute_kl_loss(pre_synergy, pre_synergy2)
        α1 = 5
        loss = ce_loss + α1 * kl_loss
        losses.update( loss.item(), len(y))
        loss.backward()
        scheduler.step()
        optimizer.step()
        if i % print_freq == 0:
            log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format( epoch=epoch, iter=i, loss=losses )
            print( log_str )         
    return losses.avg
 
def run_expriments(dataset_name, split_way, device):
    all_accuracy = np.zeros((5,1))
    all_ACC = np.zeros((5,1))
    all_BACC = np.zeros((5,1))
    all_Prec = np.zeros((5,1))
    all_Rec  = np.zeros((5,1))
    all_F1   = np.zeros((5,1))
    all_roc_auc = np.zeros((5,1))
    all_mcc = np.zeros((5,1))
    all_kappa = np.zeros((5,1))
    all_ap = np.zeros((5,1))
    n_epochs = 200

    train_loaders, valid_loaders, test_loaders =  get_data_loader_both(dataset_name, 128, split_way)
    for split in range(5):
        trainLoader = train_loaders[split]
        validLoader = valid_loaders[split]
        testLoader = test_loaders[split]
        print('数据划分完成')
        
        model = BothNet(proj_dim=512, head_num=8, dropout_rate=0.5)
        model.to(device)
        save_model_name = f'{dataset_name}_{split_way}_{split}.pt'
        earlystop = 0
        best_acc = -1.
        model = model.to(device)
        lr = 1e-3
        # 创建优化器，只包含需要更新的参数
        optimizer =AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=lr
                        )
        scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=n_epochs,
                                          steps_per_epoch=len(trainLoader))
        #记录所有epoch的结果
        ACC_s = []  # 存储每个 epoch 的准确率
        BACC_s = []
        Prec_s = []
        Rec_s = []
        F1_s = []
        roc_auc_s = []
        mcc_s = []
        kappa_s = []
        ap_s = []
        loss_s = []  # 存储每个 epoch 的损失值
        print("开始训练")
        for epochind in range(n_epochs):
            loss = train(trainLoader, model, epochind, optimizer, device, scheduler)
            accuracy,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = validate_new(validLoader, model)
            ACC_s.append(ACC)
            BACC_s.append(BACC)
            Prec_s.append(Prec)
            Rec_s.append(Rec)
            F1_s.append(F1)
            roc_auc_s.append(roc_auc)
            mcc_s.append(mcc)
            kappa_s.append(kappa)
            ap_s.append(ap)
            loss_s.append(loss)
            e_tables = PrettyTable(['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap', 'loss'])
            e_tables.float_format = '.3' 
            row = [epochind,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap, loss]
            e_tables.add_row(row)
            print(e_tables)
            # 早停止
            earlystop+=1
            if accuracy > best_acc:
                best_acc = accuracy
                save_model(model, save_model_name)
                print('best model saved!!!')
                # log_info(args, 'Best model saved, Epoch: {} | train_loss: {}, auc: {}, recall: {}, precision: {}, acc: {}'.format(i+1, loss_print, roc_auc, recall,precision, acc))
                earlystop = 0
            if earlystop > 20:
                break
        print("训练结束，开始test")
        model = load_model(model, save_model_name)
        accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap= validate_new(testLoader, model)

        e_tables = PrettyTable(['test', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap'])
        e_tables.float_format = '.3' 
        row = [epochind,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
        e_tables.add_row(row)
        print(e_tables)
        all_accuracy[split] = accuracy
        all_ACC[split] = ACC
        all_BACC[split] = BACC
        all_Prec[split] = Prec
        all_Rec[split] = Rec
        all_F1[split] = F1
        all_roc_auc[split] = roc_auc
        all_mcc[split] = mcc
        all_kappa[split] = kappa
        all_ap[split] =ap
        '''绘制训练效果曲线'''
        folder_path = f"output/{dataset_name}_{split_way}_{split}"
        if os.path.exists(folder_path): 
            # 如果存在，则删除文件夹
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), loss_s)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'loss')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/loss.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), ACC_s)
        plt.xlabel('epoch')
        plt.ylabel('ACC')
        plt.title(f'ACC')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/ACC.png')  # 保存每个图为单独的文件
        plt.show()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), BACC_s)
        plt.xlabel('epoch')
        plt.ylabel('BACC')
        plt.title(f'BACC')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/BACC.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), Prec_s)
        plt.xlabel('epoch')
        plt.ylabel('Prec')
        plt.title(f'Prec')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/Prec.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), Rec_s)
        plt.xlabel('epoch')
        plt.ylabel('Rec')
        plt.title(f'Rec')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/Rec.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), F1_s)
        plt.xlabel('epoch')
        plt.ylabel('F1')
        plt.title(f'F1')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/F1.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), roc_auc_s)
        plt.xlabel('epoch')
        plt.ylabel('roc_auc')
        plt.title(f'roc_auc')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/roc_auc.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), mcc_s)
        plt.xlabel('epoch')
        plt.ylabel('mcc')
        plt.title(f'mcc')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/mcc.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), kappa_s)
        plt.xlabel('epoch')
        plt.ylabel('kappa')
        plt.title(f'kappa')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/kappa.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        plt.figure()  # 为每个图创建一个新的图形
        plt.plot(range(0, len(loss_s)), ap_s)
        plt.xlabel('epoch')
        plt.ylabel('ap')
        plt.title(f'ap')  # 给每个图一个标题
        plt.savefig(f'{folder_path}/ap.png')  # 保存每个图为单独的文件
        plt.close()  # 显示每个图
        print("Done!")  # 训练结束
        # break

    print('*='*20)
    print('accuracy:  {0:6f}({1:6f})'.format(np.mean(all_accuracy),  np.std(all_accuracy)))
    print('ACC:  {0:6f}({1:6f})'.format(np.mean(all_ACC), np.std(all_ACC)))
    print('BACC:  {0:6f}({1:6f})'.format(np.mean(all_BACC), np.std(all_BACC)))
    print('Prec:  {0:6f}({1:6f})'.format(np.mean(all_Prec), np.std(all_Prec)))
    print('Rec:  {0:6f}({1:6f})'.format(np.mean(all_Rec), np.std(all_Rec)))
    print('F1:  {0:6f}({1:6f})'.format(np.mean(all_F1), np.std(all_F1)))
    print('roc_auc:  {0:6f}({1:6f})'.format(np.mean(all_roc_auc), np.std(all_roc_auc)))
    print('mcc:  {0:6f}({1:6f})'.format(np.mean(all_mcc), np.std(all_mcc)))
    print('kappa:  {0:6f}({1:6f})'.format(np.mean(all_kappa), np.std(all_kappa)))
    print('ap:  {0:6f}({1:6f})'.format(np.mean(all_ap), np.std(all_ap)))
    
    result_file_name = f"output/{dataset_name}_{split_way}_result.txt"
    # 打开文件以写入模式
    with open(result_file_name, "w") as file:
        file.write('ACC:  {0:.6f}({1:.6f})\n'.format(np.mean(all_ACC), np.std(all_ACC)))
        file.write('BACC:  {0:.6f}({1:.6f})\n'.format(np.mean(all_BACC), np.std(all_BACC)))
        file.write('Prec:  {0:.6f}({1:.6f})\n'.format(np.mean(all_Prec), np.std(all_Prec)))
        file.write('Rec:  {0:.6f}({1:.6f})\n'.format(np.mean(all_Rec), np.std(all_Rec)))
        file.write('F1:  {0:.6f}({1:.6f})\n'.format(np.mean(all_F1), np.std(all_F1)))
        file.write('roc_auc:  {0:.6f}({1:.6f})\n'.format(np.mean(all_roc_auc), np.std(all_roc_auc)))
        file.write('mcc:  {0:.6f}({1:.6f})\n'.format(np.mean(all_mcc), np.std(all_mcc)))
        file.write('kappa:  {0:.6f}({1:.6f})\n'.format(np.mean(all_kappa), np.std(all_kappa)))
        file.write('ap:  {0:.6f}({1:.6f})\n'.format(np.mean(all_ap), np.std(all_ap)))


import argparse
if __name__ == "__main__":

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run experiments with specified dataset and split way.")
    
    # 添加命令行参数
    parser.add_argument('--dataset_name', type=str, default='drugcomb', help='Name of the dataset (e.g., drugcomb, drugcombdb)')
    parser.add_argument('--split_way', type=str, default='random', help='Split way (e.g., random, cell, drugs, drug)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_name = args.dataset_name  #drugcomb, drugcombdb
    split_way = args.split_w   #random, cell, drugs, drug
    run_expriments(dataset_name=dataset_name, split_way=split_way, device=device)