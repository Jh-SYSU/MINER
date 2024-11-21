import torch
import argparse
from databuilder.mergeDataset import mergeDataset
from torch_geometric.loader import DataLoader
from model.MINER import MINER
import torch.optim as optim
import os
from tqdm import tqdm
from utils.metrics import cal_rocauc, cal_aup
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import json
import random
from torch import nn
import warnings
warnings.filterwarnings("ignore")

def set_json(json_file, parser):
    
    with open(json_file, 'r') as file:
        args_data = json.load(file)
        
    for key, value in args_data.items():
        setattr(parser, key, value)
        
    return parser

def train(model, loader, device, optimizer, alpha, beta, cigmma, gamma, weight):
    model.train()
    loss_sum = 0
    loss_cvc = 0
    loss_ic = 0
    loss_corr = 0
    loss_classifier = 0
    loss_recon = 0
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        _, loss, ic, cvc, classifier, corr, recon= model(batch, alpha, beta, cigmma, gamma, weight)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        loss_sum += loss
        loss_cvc += cvc
        loss_ic += ic
        loss_classifier += classifier
        loss_corr += corr
        loss_recon += recon
    return loss_sum, loss_ic, loss_cvc, loss_classifier, loss_corr, loss_recon

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _, _, _, _, _, _= model(batch, 0, 0, 0, 0, 0)
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    
    return cal_rocauc(y_true, y_pred), cal_aup(y_true, y_pred)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epoch", type=int, default=150)
    
    parser.add_argument("--dnn_out_dim", type=int, default=256)
    parser.add_argument("--dnn_layer", type=int, default=5)
    parser.add_argument("--de_gene_dnn_emb", type=int, default=978//2)
    parser.add_argument("--de_cell_dnn_emb", type=int, default=1783//2)
    parser.add_argument("--gene_dnn_emb", type=int, default=978//2)
    parser.add_argument("--cell_dnn_emb", type=int, default=1783//2)
    
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--gnn_layer", type=int, default=3)
    parser.add_argument("--gnn_out_dim", type=int, default=128)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--drop_ratio", type=float, default=0.2)
    
    parser.add_argument("--unuse_compound", action="store_true")
    parser.add_argument("--unuse_gene", action="store_true")
    parser.add_argument("--unuse_cell", action="store_true")
    
    parser.add_argument("--alpha", type=float, help="correlation loss", default=0.1)
    parser.add_argument("--beta", type=float, help="contrastive loss", default=0.1)
    parser.add_argument("--sigma", type=float, help="recon loss", default=0.05)
    parser.add_argument("--gamma", type=float, help="balance loss", default=0.01)
    parser.add_argument("--weight", type=float,  default=0)
    
    parser.add_argument("--mmc", type=str, default="MINER")
    parser.add_argument("--dataset", type=str, default="broad6k")
    
    args = parser.parse_args()
    
    seeds = [0, 10, 20, 30, 40]
    
    root = "./dataset"
    root = os.path.abspath(root)
    
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() 
                                                    else torch.device('cpu'))
    # device = torch.device('cpu')     
    print(device)                     
    if args.dataset == 'broad6k':
        args = set_json("./args_json/broad6k.json", args)
        data_files = {
                    "assay_data":"assays.csv.gz", 
                    "mo": "CP-Bray.csv.gz",
                    "mo_feature": "CP-Bray_feature.npz",
                    "ge": "GE.csv.gz",
                    "ge_feature": "GE_feature.npz"
                    }
        num_tasks = 32
        dataset = mergeDataset(root=root, dataset_name=args.dataset, num_tasks=num_tasks, data_file=data_files)
        split_idx = dataset.get_idx_split(split_type="scaffold")
        
    elif args.dataset == 'chembl2k':
        parser = set_json("./args_json/chembl2k.json", args)
        data_files = {
                    "assay_data":"assays.csv.gz", 
                    "mo": "CP-JUMP.csv.gz",
                    "mo_feature": "CP-JUMP_feature.npz", 
                    "ge": "GE.csv.gz", 
                    "ge_feature": "GE_feature.npz"
                }
        num_tasks = 41
        dataset = mergeDataset(root=root, dataset_name=args.dataset, num_tasks=num_tasks, data_file=data_files)
        split_idx = dataset.get_idx_split(split_type="scaffold")

    data_frame = pd.read_csv(os.path.join(root, args.dataset, "assay_data", "assays.csv"))
    col_name = data_frame.columns[-num_tasks:]
    col_name = list(col_name)
    col_name.append("mean")
    
    log = "./log/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log = os.path.abspath(log)
    os.mkdir(log)
    params = vars(args)  
    with open(os.path.join(log, "log.json"), 'w') as json_file:
        json.dump(params, json_file, indent=4)

    f_auc = open(os.path.join(log, "auc"+".csv"), "w")
    f_aup = open(os.path.join(log, "aup"+".csv"), "w")
    writer_auc = csv.writer(f_auc)
    writer_aup = csv.writer(f_aup)
    writer_auc.writerow(col_name)
    writer_aup.writerow(col_name)
    
    input_modal = []
    if not args.unuse_compound:
        input_modal.append("compound")
    if not args.unuse_gene:
        input_modal.append("gene")
    if not args.unuse_cell:
        input_modal.append('cell')
    best_auc_test = []
    best_aup_test = []
    auc_90_num = []
    auc_85_num = []
    auc_80_num = []
    
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        
        data_frame = pd.read_csv(os.path.join(root, args.dataset, "assay_data", "assays.csv"))
        col_name = data_frame.columns[-num_tasks:]
        col_name = list(col_name)
        col_name.append("mean")
        
        log_file = os.path.join(log, args.dataset + "_" + str(seed))
        os.makedirs(log_file, exist_ok=True)
        
        
        best_auc_test_each = []
        best_aup_test_each = None
        
        lr = args.lr 
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        mo_dim = dataset[0].mo.shape[1]
        ge_dim = dataset[0].ge.shape[1]
        num_tasks = dataset[0].y.shape[1]
        
        if args.mmc == "MINER":
            model = MINER(fingerprint_num=300, input_modal=input_modal, gene_feature_num=ge_dim, cell_profile_feature_num=mo_dim, 
                            target_num=num_tasks, gnn_layer=args.gnn_layer, gcn_out_dim=args.gnn_out_dim, dnn_layer=args.dnn_layer, 
                            dnn_out_dim=args.dnn_out_dim, drop_ratio=args.drop_ratio, emb_size=args.emb_size, gnn_type=args.gnn_type,
                            de_gene_dnn_emb=args.de_gene_dnn_emb, de_cell_dnn_emb=args.de_cell_dnn_emb,
                            gene_dnn_emb=args.gene_dnn_emb, cell_dnn_emb=args.cell_dnn_emb)  

        model.to(device)
        
        f_log = open(os.path.join(log_file, "log"+".txt"), "w")
        f_log.write(f"epoch {args.epoch}, lr {lr}\n")
        
        best = [0, 0, 0, 0]
        alpha = args.alpha
        beta = args.beta
        sigma = args.sigma
        gamma = args.gamma
        weight = args.weight
        
        if args.dataset == "broad6k":
            optimizer = optim.Adam(model.parameters(), lr=lr) 
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.0)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=[0.85, 0.85]) 
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
            
        for epoch in range(1, args.epoch + 1):
            
            alpha = args.alpha
            beta = args.beta
            sigma = args.sigma
            
            print("=====Epoch {}".format(epoch))
            print('Training...')
            loss = train(model, train_loader, device, optimizer, alpha, beta, sigma, gamma, weight)
            
            print('Evaluating...')
            
            train_auc, train_aup = eval(model, device, train_loader)
            val_auc, val_aup = eval(model, device, val_loader)
            test_auc, test_aup = eval(model, device, test_loader)
            
            scheduler.step()
            
            print("lr:", optimizer.param_groups[0]['lr'])
            print("loss:", loss)
            f_log.write(f"epoch {epoch} lr {optimizer.param_groups[0]['lr']} alpha {alpha} beta {beta} sigma {sigma} gamma {gamma}\nloss {loss}\n")
            
            train_valid_auc = train_auc[train_auc == train_auc]
            val_valid_auc = val_auc[val_auc == val_auc]
            test_valid_auc = test_auc[test_auc == test_auc]
            print('auc Train:', train_valid_auc.mean(), 'Val:', val_valid_auc.mean())
            f_log.write(f"auc Train {train_valid_auc.mean()} val {val_valid_auc.mean()}\n")
            
            
            train_valid_aup = train_aup[train_aup == train_aup]
            val_valid_aup = val_aup[val_aup == val_aup]
            test_valid_aup = test_aup[test_aup == test_aup]
            print('aup Train:', train_valid_aup.mean(), 'Val:',val_valid_aup.mean())
            f_log.write(f"aup Train {train_valid_aup.mean()} val {val_valid_aup.mean()} \n\n")
            
            tmp = val_valid_auc.mean()
            if tmp > best[0]:
                best[0] = tmp
                best[1] = val_valid_aup.mean()
                best[2] = test_valid_auc.mean()
                best[3] = test_valid_aup.mean()
                best_auc_test_each = test_auc
                best_aup_test_each = test_aup
                torch.save(model.state_dict(), os.path.join(log_file, "auc"+".pth"))
            print("best test auc:", best[2], "best test aup:", best[3])
            print("best test auc:", test_valid_auc.mean(), "best test aup:", best[3])
            test_auc = np.append(test_auc, test_valid_auc.mean())
            test_aup = np.append(test_aup, test_valid_aup.mean())
            f_log.flush()
            
        best_auc_test.append(best[2])
        best_aup_test.append(best[3])
        print("seed:", seed, "\nbest test auc:", best[2], " best test aup:", best[3])
        
        f_log.write(f"best test auc:{best[2]}\t best test aup:{best[3]}\n")

        
        writer_auc.writerow(best_auc_test_each)
        writer_aup.writerow(best_aup_test_each)
        f_auc.flush()
        f_aup.flush()
        auc_80_num.append((sum(1 for x in best_auc_test_each[:-1] if x >= 0.8)) / num_tasks)
        auc_85_num.append((sum(1 for x in best_auc_test_each[:-1] if x >= 0.85)) / num_tasks)
        auc_90_num.append((sum(1 for x in best_auc_test_each[:-1] if x >= 0.9)) / num_tasks) 
        print(f"seed:{seed}\nauc>80%:{auc_80_num[-1]}\t auc>85%:{auc_85_num[-1]}\t auc>90%:{auc_90_num[-1]}\n")
        f_log.write(f"auc>80%:{auc_80_num[-1]}\t auc>85%:{auc_85_num[-1]}\t auc>90%:{auc_90_num[-1]}\n")
        f_log.flush()
        
    tmp = [best_auc_test, best_aup_test, auc_80_num, auc_85_num, auc_90_num]
    tmp_str = []
    for i in range(5):
        tmp_str.append(np.mean(tmp[i]))
        tmp_str.append(np.std(tmp[i]))
        
    params['auc_test_mean'] = tmp_str[0]
    params['auc_test_std'] = tmp_str[1]
    params['aup_test_mean'] = tmp_str[2]
    params['aup_test_std'] = tmp_str[3]
    params['auc_80_num_mean'] = tmp_str[4]
    params['auc_80_num_std'] = tmp_str[5]
    params['auc_85_num_mean'] = tmp_str[6]
    params['auc_85_num_std'] = tmp_str[7]
    params['auc_90_num_mean'] = tmp_str[8]
    params['auc_90_num_std'] = tmp_str[9]
    
    params['test_best'] = tmp
    with open(os.path.join(log, "log.json"), 'w') as json_file:
        json.dump(params, json_file, indent=4)
        
    print(f"auc:{tmp_str[0]}±{tmp_str[1]} \t aup:{tmp_str[2]}±{tmp_str[3]}")
    print(f"auc_80:{tmp_str[4]}±{tmp_str[5]} \t auc_85:{tmp_str[6]}±{tmp_str[7]}\t auc_90:{tmp_str[8]}±{tmp_str[9]}")
    f_log.close()        
    
    f_aup.close() 
       
    print("finish")
                
                