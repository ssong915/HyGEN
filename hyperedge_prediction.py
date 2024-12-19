import os
import time

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
from torchmetrics import AveragePrecision

import models
import utils
from aggregator import *
from data_load import gen_DGLGraph, gen_data, load_test, load_train, load_val
from hygenerator import HyGenerator
from sampler import *
from training import model_eval, model_train

def train(args):            
    os.makedirs(f"{args.curr}/data/checkpoints/{args.dataset_name}/{args.folder_name}/{args.random}", exist_ok=True)
    os.makedirs(f"{args.curr}/data/logs/{args.dataset_name}/{args.folder_name}/{args.random}", exist_ok=True)
    f_log = open(f"{args.curr}/data/logs/{args.dataset_name}/{args.folder_name}/{args.random}/{args.random}_train.log", "w")
    f_log.write(f"args: {args}\n")
    
    if args.fix_seed:
        np.random.seed(0)
        torch.manual_seed(0)
        
    train_DG = "epoch1:1".split(":")
    args.train_DG = [int(train_DG[0][5:]), int(train_DG[1]), int(train_DG[0][5:])+int(train_DG[1])]    
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    DATA = args.dataset_name    
    
    for j in tqdm(range(args.exp_num)):  
                  
        # Load data
        args = gen_data(args, args.dataset_name)
        data_dict = torch.load(f"{args.curr}/data/splits/{args.dataset_name}split{j}.pt")
        ground = data_dict["ground_train"] + data_dict["ground_valid"]
        g = gen_DGLGraph(args, ground)  
        train_batchloader = load_train(data_dict, args.bs, device) # only positives
        val_batchloader_pos = load_val(data_dict, args.bs, device, label="pos")
        val_batchloader_sns = load_val(data_dict, args.bs, device, label="sns")
        val_batchloader_mns = load_val(data_dict, args.bs, device, label="mns")
        val_batchloader_cns = load_val(data_dict, args.bs, device, label="cns")
    
        # Initialize models
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                       args.n_layers, memory_dim=args.nv)
        model.to(device)        
        Aggregator = None    
        cls_layers = [args.dim_vertex, 128, 8, 1]
        Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
        Aggregator.to(device)  
        size_dist = utils.gen_size_dist(ground)
               
        Generator =  HyGenerator(args, img_size=args.dim_edge, noise_dim = args.noise_dim, device=device, size_dist=size_dist)
            
        Generator.to(device)
        average_precision = AveragePrecision(task='binary')

        best_roc = 0
        best_epoch = 0 
        optim_D = torch.optim.RMSprop(list(model.parameters())+list(Aggregator.parameters()), lr=args.D_lr)
        optim_G = torch.optim.RMSprop(Generator.parameters(), lr=args.G_lr)
        
        pos_size_total = 0
        neg_size_total = 0    
        
        for epoch in tqdm(range(args.epochs), leave=False):   
            train_pred, train_label = [], []
            pos_onehot_list, neg_onehot_list = [], []
            
            d_loss_sum, g_loss_sum, count  = 0.0, 0.0, 0
            
            print("--------------------epoch start: ", epoch)
            epoch_time = 0.0
            epoch_start_time = time.time()
            
            # Train
            while True :
                pos_hedges, pos_labels, is_last = train_batchloader.next()
                d_loss, g_loss, train_pred, train_label, pos_onehot, neg_onehot = model_train(args, g, model, Aggregator, Generator, optim_D, optim_G, pos_hedges, pos_labels, train_pred, train_label, device, epoch)

                d_loss_sum += d_loss
                g_loss_sum += g_loss
                
                pos_onehot_list.extend(pos_onehot)
                tmp_neg_list = [neg.tolist() for neg in neg_onehot]
                neg_onehot_list.extend(tmp_neg_list)
                count += 1
                if is_last :
                    break

            train_pred = torch.stack(train_pred)
            train_pred = train_pred.squeeze()
        
            #------------- modification -------------#
            train_label = torch.round(torch.cat(train_label, dim=0))
            train_label = train_label.type(torch.int64)
            #-----------------------------------------#
                    
            train_roc = metrics.roc_auc_score(np.array(train_label.cpu()), np.array(train_pred.cpu()))
            train_ap = average_precision(torch.tensor(train_pred), torch.tensor(train_label))            
            
            f_log.write(f'{epoch} epoch: Training d_loss : {d_loss_sum / count} / Training g_loss : {g_loss_sum / count} /')
            f_log.write(f'Training roc : {train_roc} / Training ap : {train_ap} \n')

            val_pred_pos, total_label_pos, _, _ = model_eval(args, val_batchloader_pos, g, model, Aggregator,device)
            val_pred_sns, total_label_sns, _, _ = model_eval(args, val_batchloader_sns, g, model, Aggregator,device)
            auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, val_pred_pos+val_pred_sns)
            f_log.write(f"{epoch} epoch, SNS : Val AP : {ap_sns} / AUROC : {auc_roc_sns}\n")

            val_pred_mns, total_label_mns, _, _ = model_eval(args, val_batchloader_mns, g, model, Aggregator,device)
            auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, val_pred_pos+val_pred_mns)
            f_log.write(f"{epoch} epoch, MNS : Val AP : {ap_mns} / AUROC : {auc_roc_mns}\n")

            val_pred_cns, total_label_cns, _, _ = model_eval(args, val_batchloader_cns, g, model, Aggregator,device)
            auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, val_pred_pos+val_pred_cns)
            f_log.write(f"{epoch} epoch, CNS : Val AP : {ap_cns} / AUROC : {auc_roc_cns}\n")

            l = len(val_pred_pos)//3
            val_pred_all = val_pred_pos + val_pred_sns[0:l] + val_pred_mns[0:l] + val_pred_cns[0:l]
            total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
            auc_roc_all, ap_all = utils.measure(total_label_all, val_pred_all)
            f_log.write(f"{epoch} epoch, ALL : Val AP : {ap_all} / AUROC : {auc_roc_all}\n")
            f_log.flush()
            
            # Save best checkpoint
            if best_roc < (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3:
                best_roc = (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3
                best_epoch=epoch
                no_improvement_count = 0          
                torch.save(model.state_dict(), f"{args.curr}/data/checkpoints/{args.dataset_name}/{args.folder_name}/{args.random}/model_{j}.pkt")
                torch.save(Aggregator.state_dict(), f"{args.curr}/data/checkpoints/{args.dataset_name}/{args.folder_name}/{args.random}/Aggregator_{j}.pkt")
                torch.save(Generator.state_dict(), f"{args.curr}/data/checkpoints/{args.dataset_name}/{args.folder_name}/{args.random}/Generator_{j}.pkt")
            else:
                no_improvement_count += 1
                if no_improvement_count >= args.patience:                     
                    break                     
        
        with open(f"{args.curr}/data/checkpoints/{args.dataset_name}/{args.folder_name}/{args.random}/best_epochs.logs", "a") as e_log:  
            e_log.write(f"exp {j} best epochs: {best_epoch}\n")

        test_batchloader_pos = load_test(data_dict, args.bs, device, label="pos")

    f_log.close()
    
    return args

def test(args, j):    
    args.checkpoint = f"{args.curr}/data/checkpoints/{args.dataset_name}/{args.folder_name}/{args.random}" 
    os.makedirs(f"{args.curr}/data/logs/{args.dataset_name}/{args.random}", exist_ok=True)
    f_log = open(f"{args.curr}/data/logs/{args.dataset_name}/{args.folder_name}/{args.random}_results.log", "w")    
    f_log.write(f"{args}\n")    
    
    # Load data
    data_dict = torch.load(f"{args.curr}/data/splits/{args.dataset_name}split{j}.pt")
    args = gen_data(args, args.dataset_name)
    ground = data_dict["ground_train"] + data_dict["ground_valid"]
    g = gen_DGLGraph(args, ground)
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

    # test set    
    test_batchloader_pos = load_test(data_dict, args.bs, device, label="pos")
    test_batchloader_sns = load_test(data_dict, args.bs, device, label="sns")
    test_batchloader_mns = load_test(data_dict, args.bs, device, label="mns")
    test_batchloader_cns = load_test(data_dict, args.bs, device, label="cns")

    # Initialize models
    model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                    args.n_layers, memory_dim=args.nv, K=args.memory_size)
    model.to(device)
    if args.memory_size == 0:
        checkpoint = torch.load(f"{args.checkpoint}/model_{j}.pkt")
        if "memory" in checkpoint:
            del checkpoint["memory"]
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(f"{args.checkpoint}/model_{j}.pkt"))
        
    cls_layers = [args.dim_vertex, 128, 8, 1]
    Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
    Aggregator.to(device)
    Aggregator.load_state_dict(torch.load(f"{args.checkpoint}/Aggregator_{j}.pkt"))
    
    model.eval()
    Aggregator.eval()

    with torch.no_grad():
        test_pred_pos, total_label_pos, test_pos_embeddings, test_pos_hedges = model_eval(args, test_batchloader_pos, g, model, Aggregator, device)
        test_pred_sns, total_label_sns, test_sns_embeddings, test_sns_hedges = model_eval(args, test_batchloader_sns, g, model, Aggregator, device)
        auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, test_pred_pos+test_pred_sns)
        f_log.write(f"SNS : Test AP : {ap_sns} / AUROC : {auc_roc_sns}\n")

        test_pred_cns, total_label_cns, test_cns_embeddings, test_cns_hedges = model_eval(args, test_batchloader_cns, g, model, Aggregator, device)
        auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, test_pred_pos+test_pred_cns)
        f_log.write(f"CNS : Test AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
        
        test_pred_mns, total_label_mns, test_mns_embeddings, test_mns_hedges = model_eval(args, test_batchloader_mns, g, model, Aggregator, device)
        auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, test_pred_pos+test_pred_mns)
        f_log.write(f"MNS : Test AP : {ap_mns} / AUROC : {auc_roc_mns}\n")
        
        l = len(test_pred_pos)//3
        test_pred_all = test_pred_pos + test_pred_sns[0:l] + test_pred_mns[0:l] + test_pred_cns[0:l]
        total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
        auc_roc_all, ap_all = utils.measure(total_label_all, test_pred_all)
        f_log.write(f"ALL : Test AP : {ap_all} / AUROC : {auc_roc_all}\n")
        result = {'ap_sns':ap_sns, 'auc_roc_sns':auc_roc_sns, 'ap_mns':ap_mns, 'auc_roc_mns':auc_roc_mns, 'ap_cns':ap_cns, 'auc_roc_cns':auc_roc_cns, 'ap_all':ap_all, 'auc_roc_all':auc_roc_all}
        f_log.flush()
        
        return result        
        
if __name__ == "__main__":
    args = utils.parse_args()

    args.curr = os.getcwd()
    args.folder_name ='exp1'
    print(args.dataset_name) 
    print(args.folder_name)
    
    ap_sns_list = []
    auc_roc_sns_list = []
    ap_mns_list = []
    auc_roc_mns_list = []
    ap_cns_list = []
    auc_roc_cns_list = []
    ap_all_list = []
    auc_roc_all_list = []

    train(args)
    for j in range(args.exp_num):
        result = test(args, j)
    
    ap_sns_list.append(result['ap_sns'])
    auc_roc_sns_list.append(result['auc_roc_sns'])
    ap_mns_list.append(result['ap_mns'])
    auc_roc_mns_list.append(result['auc_roc_mns'])
    ap_cns_list.append(result['ap_cns'])
    auc_roc_cns_list.append(result['auc_roc_cns'])
    ap_all_list.append(result['ap_all'])
    auc_roc_all_list.append(result['auc_roc_all'])

    final_result = {
        'ap_sns': np.mean(ap_sns_list),
        'auc_roc_sns': np.mean(auc_roc_sns_list),
        'ap_mns': np.mean(ap_mns_list),
        'auc_roc_mns': np.mean(auc_roc_mns_list),
        'ap_cns': np.mean(ap_cns_list),
        'auc_roc_cns': np.mean(auc_roc_cns_list),
        'ap_all': np.mean(ap_all_list),
        'auc_roc_all': np.mean(auc_roc_all_list)
    }
    
    print('ap_sns_list\t ap_mns_list\t ap_cns_list\t ap_all_list\t ')
    print(f'{np.mean(ap_sns_list):.4f}\t{np.mean(ap_mns_list):.4f}\t{np.mean(ap_cns_list):.4f}\t{np.mean(ap_all_list):.4f}')
    print('auc_roc_sns_list\t auc_roc_mns_list\t auc_roc_cns_list\t auc_roc_all_list\t ')
    print(f'{np.mean(auc_roc_sns_list):.4f}\t{np.mean(auc_roc_mns_list):.4f}\t{np.mean(auc_roc_cns_list):.4f}\t{np.mean(auc_roc_all_list):.4f}')

    print(final_result)

    