import torch.nn as nn
from model.dnn import DNN as DNN
from model.gnn import GNN
import torch
from model.MoE import MoE
from model.MoE import MoE

_EPSILON = 1e-8

def div(x_, y_):
    return torch.div(x_, y_ + _EPSILON)

def log(x_):
    return torch.log(x_ + _EPSILON)


class MINER(nn.Module):
    def __init__(self, input_modal, fingerprint_num, gene_feature_num, cell_profile_feature_num, target_num, 
                 gnn_layer, gcn_out_dim, dnn_layer, dnn_out_dim, de_gene_dnn_emb, de_cell_dnn_emb,
                 gene_dnn_emb, cell_dnn_emb,emb_size=256,drop_ratio=0.3, gnn_type="gin"):
        super().__init__()
        self.use_compound = ('compound' in input_modal) + 0
        self.use_gene = ('gene' in input_modal) + 0
        self.use_cell = ('cell' in input_modal) + 0
        self.num_modal = self.use_compound + self.use_gene + self.use_cell
        

        # encoder, generate common, specific message
        if self.use_compound:
            self.gnn = GNN(gnn_type=gnn_type, emb_dim=gcn_out_dim, num_layer=gnn_layer, num_tasks=target_num, drop_ratio=drop_ratio, virtual_node=False)
            self.gnn.graph_pred_linear = nn.Identity()
            self.dnn_compound = DNN(input_dim=fingerprint_num, output_dim=dnn_out_dim, num_layer=dnn_layer, emb_dim=2048, drop_ratio=drop_ratio)
            self.fc1_pil = nn.Sequential(
                                        nn.Linear(gcn_out_dim, emb_size), 
                                        )
            self.fc1_cil = nn.Sequential(
                                        nn.Linear(gcn_out_dim, emb_size), 
                                        )
            self.fc1_pil_ = nn.Sequential(
                                        nn.Linear(dnn_out_dim, emb_size), 
                                        )
            self.fc1_cil_ = nn.Sequential(
                                        nn.Linear(dnn_out_dim, emb_size), 
                                        )
        if self.use_gene:
            self.dnn = DNN(input_dim=gene_feature_num, output_dim=dnn_out_dim, num_layer=dnn_layer, emb_dim=gene_dnn_emb, drop_ratio=drop_ratio)
            self.fc2_pil = nn.Linear(dnn_out_dim, emb_size)
            self.fc2_cil = nn.Linear(dnn_out_dim, emb_size)
            
        if self.use_cell:
            self.cell_dnn = DNN(input_dim=cell_profile_feature_num, output_dim=dnn_out_dim, num_layer=dnn_layer, emb_dim=cell_dnn_emb, drop_ratio=drop_ratio)
            self.fc3_pil = nn.Linear(dnn_out_dim, emb_size)
            self.fc3_cil = nn.Linear(dnn_out_dim, emb_size)
        
        
        # decoder
        if self.use_gene:
            self.de_gene_dnn = DNN(input_dim=emb_size, output_dim=gene_feature_num, num_layer=dnn_layer, emb_dim=de_gene_dnn_emb, drop_ratio=drop_ratio)
            
        if self.use_cell:
            self.de_cell_dnn = DNN(input_dim=emb_size, output_dim=cell_profile_feature_num, num_layer=dnn_layer, emb_dim=de_cell_dnn_emb, drop_ratio=drop_ratio)

         
        self.MoE_layer_com = MoE(emb_size, emb_size, 4, 4)
        
        # predictor
        self.output_block = nn.Linear(emb_size, target_num)
         
         
    
    def cal_loss_cont(self, x_com, x_pri, weight):
        loss_cvc = 0
         
        for i in range(self.num_modal):
            x_com[i] = x_com[i] / x_com[i].norm(dim = -1, keepdim = True)    
    
        n = 0
        for i in range(self.num_modal):
            for j in range(self.num_modal):
                if i != j and (i == 0 or j == 0):
                    matrix = torch.matmul(x_pri[i], x_pri[j].T)
                    matrix = torch.sigmoid(matrix)
                    loss_cvc += self.bunsupLoss(x_com[i], x_com[j], matrix, weight)
                    n += 1
        if n != 0:
            loss_cvc /= n
        
        return loss_cvc
    
    def cal_loss_corr(self, x_pri, x_com):
        loss_ic = 0
        for pri, com in zip(x_pri, x_com):
            mean_pri = torch.mean(pri, dim=1, keepdim=True)
            mean_com = torch.mean(com, dim=1, keepdim=True)
            pri = pri - mean_pri
            com = com - mean_com
            numerator = torch.sum(pri * com, dim=1)
            denominator = torch.sqrt(torch.sum(pri**2, dim=1) * torch.sum(com**2, dim=1))
            correlation_coeffs = numerator / denominator
            loss_ic += correlation_coeffs / self.num_modal
        
        loss_ic = torch.exp(loss_ic)
        loss_ic = loss_ic.mean()
        
        return loss_ic

    def bunsupLoss(self, feature_a, feature_b, scores, weight=0, temperature=0.07):
        scores = scores.detach()
        similarity = torch.einsum('nc,mc->nm', [feature_a, feature_b]) / temperature 
        posmask = (torch.ones_like(similarity) - torch.eye(similarity.shape[0]).to(similarity.device))
        loss = -torch.log(torch.exp(torch.diagonal(similarity) * (1-scores/10)) / (torch.exp(similarity * (1-scores * weight)) * posmask).sum(1))
        mean_loss = loss.mean()
        
        return mean_loss
    
    def forward(self, batch, alpha, beta, sigma, gamma, weight):
        # loss_tr = FocalLoss(gamma=1, alpha=1)
        loss_tr = nn.BCEWithLogitsLoss()
        labels = batch.y
        masks = labels == labels
        
        x_pri = []
        x_com = []
        x_pri_ = []
        x_com_ = []
        ge_features = batch.ge
        cell_imgs = batch.mo

        if self.use_compound:
            x1 = self.gnn(batch)
            x1_ = self.dnn_compound(batch.fingerprint)
            x1_pri = self.fc1_pil(x1)
            x1_com = self.fc1_cil(x1)
            x1_pri_ = self.fc1_pil_(x1_)
            x1_com_ = self.fc1_cil_(x1_)
            x_pri.append(x1_pri)
            x_com.append(x1_com)
            x_pri_.append(x1_pri_)
            x_com_.append(x1_com_)
            
            # decoder
            if self.use_gene:
                de_gene = self.de_gene_dnn(x1_com)
                ge_label = batch.ge_label == 1
                for i in range(len(ge_label)):
                    if ge_label[i] == 0:
                        ge_features[i] = de_gene[i]
                        
            if self.use_cell:
                de_mo = self.de_cell_dnn(x1_com)
                mo_label = batch.mo_label == 1
                for i in range(len(mo_label)):
                    if mo_label[i] == 0:
                        cell_imgs[i] = de_mo[i]   
            
        if self.use_gene:
            x2 = self.dnn(ge_features)
            x2_pri = self.fc2_pil(x2)
            x2_com = self.fc2_cil(x2)
            x_pri.append(x2_pri)
            x_com.append(x2_com)
            x_pri_.append(x2_pri)
            x_com_.append(x2_com)
            
        if self.use_cell:
            x3 = self.cell_dnn(cell_imgs)
            x3_pri = self.fc3_pil(x3)
            x3_com = self.fc3_cil(x3)
            x_pri.append(x3_pri)
            x_com.append(x3_com)
            x_pri_.append(x3_pri)
            x_com_.append(x3_com)
            
        
            
        loss_corr = 0
        loss_cont = 0
        loss_recon = 0
        balance_loss = 0
        
        #  Pearson’s correlation coefficient
        if alpha != 0:
            loss_corr = self.cal_loss_corr(x_pri, x_com) + self.cal_loss_corr(x_pri_, x_com_)
        else:
            loss_corr = 0
        
        
        # recon loss
        if sigma == 0:
            loss_recon = 0
        else:
            mse_loss = nn.MSELoss()
            if self.use_gene:
                loss_ge = mse_loss(ge_features, de_gene) / sum(ge_label) 
            else:
                loss_ge = 0
            
            if self.use_cell:
                loss_mo = mse_loss(cell_imgs, de_mo) / sum(mo_label)
            else:
                loss_mo =0
            loss_recon = loss_ge + loss_mo
        
        x_com.append(x1_com_)
        x, com_weight= self.MoE_layer_com(x_com)
        
        
        if gamma != 0:
            balance_loss = torch.std(com_weight.sum(0)) 
        else:
            balance_loss = 0
            
        # contrastive loss 
        if beta != 0: 
            loss_cont = self.cal_loss_cont(x_com, x_pri, weight) + self.cal_loss_cont(x_com_, x_pri_, weight)
        else:
            loss_cont = 0
            
        preds = self.output_block(x)
        
        loss_classifier = loss_tr(preds[masks], labels[masks])
        
        loss = alpha * loss_corr + beta * loss_cont + sigma * loss_recon + gamma * balance_loss + loss_classifier
        preds = torch.sigmoid(preds)
        
        return preds, loss, loss_corr, loss_cont, loss_recon, balance_loss, loss_classifier
 