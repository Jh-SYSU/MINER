import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
import os
import csv
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np 
import deepchem as dc

# for three multimodality

if __name__ == "__main__":
    from utils.feature import smileToList
    from utils.split_data import random_split, scaffold_split
    from utils.scaler import StandardScaler
else:
    from databuilder.utils.feature import smileToList
    from databuilder.utils.split_data import random_split, scaffold_split
    from databuilder.utils.scaler import StandardScaler

class mergeDataset(InMemoryDataset):
    
    def __init__(self, root: str, dataset_name: str, data_file: dict, task_type="classification", num_tasks = 41, transform=None, pre_transform=None, pre_filter=None):
        # the file name that saved assay data, 
        self.datasetname = dataset_name
        self.data_file = data_file
        self.num_tasks = num_tasks
        self.task_type = task_type
        super(mergeDataset, self).__init__(root=os.path.join(root, self.datasetname), transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    @property
    def raw_file_names(self):
        return []   
    
    @property
    def processed_file_names(self):
        return ['mergeDataset.pt']
    
    def make_compound_feature_paris(self, data_df, feature_df, feature_arr):
        x_list = []
        feature_dim = feature_arr.shape[1]
        for _, row in data_df.iterrows():
            if len(feature_df[feature_df["inchikey"] == row["inchikey"]]) == 0:
                x_list.append(np.array([float("nan")] * feature_dim))
            else:
                x_tensor = np.array(
                    feature_arr[
                        feature_df[
                            feature_df["inchikey"] == row["inchikey"]
                        ].index.tolist()[0]
                    ],
                    dtype=np.float32,
                )
                x_list.append(x_tensor)
        x = np.stack(x_list)
        
        return x
    
    def process(self):
        # make two directories: assay_data for storing smiles and corresponding assays, feature_data for storing gene and image
        os.makedirs(os.path.join(self.root, "assay_data"), exist_ok=True)
        if not os.path.exists(os.path.join(self.root, "assay_data", "assays.csv")):
            df = pd.read_csv(
                os.path.join(self.root, "raw", self.data_file['assay_data']),
                compression="gzip"
                )
            df.to_csv(os.path.join(self.root, "assay_data", "assays.csv"), index=False)
        
        
        
        os.makedirs(os.path.join(self.root, "feature_data"), exist_ok=True)
        if not os.path.exists(os.path.join(self.root, "feature_data", "ge.npz")):

            ge_df = pd.read_csv(
                os.path.join(self.root, "raw", self.data_file['ge']), 
                 compression="gzip")
            ge_features = np.load(os.path.join(self.root, "raw", self.data_file['ge_feature']))['data']
            ge_features = self.make_compound_feature_paris(cs_df, ge_df, ge_features)
            np.savez(os.path.join(self.root, "feature_data", "ge.npz"), features=ge_features)
        
        if not os.path.exists(os.path.join(self.root, "feature_data", "mo.npz")):
            cs_df = pd.read_csv(
                os.path.join(self.root, "raw", self.data_file['assay_data']),
                compression="gzip"
                )
            mo_df = pd.read_csv(
                os.path.join(self.root, "raw", self.data_file['mo']), 
                 compression="gzip")
            mo_features = np.load(os.path.join(self.root, "raw", self.data_file['mo_feature']))['data']
            mo_features = self.make_compound_feature_paris(cs_df, mo_df, mo_features)
            np.savez(os.path.join(self.root, "feature_data", "mo.npz"), features=mo_features)
        
        
        # read smiles and labels
        df = pd.read_csv(os.path.join(self.root, "assay_data", "assays.csv"))
        # smiles
        smiles = df["smiles"].tolist()
        featurizer = dc.feat.Mol2VecFingerprint()
        # featurizer = dc.feat.MorganGenerator(radius=2, nBits=2048)
        features = torch.tensor(featurizer.featurize(smiles))
        scaler_drug = StandardScaler(mean=torch.from_numpy(np.mean(np.array(list(features)), 0)),
                                    std=torch.from_numpy(np.std(np.array(list(features)), 0)))
        features = [torch.clip(scaler_drug.transform(f), -2.5, 2.5) for f in features]
        
        # labels
        labels_name = df.columns[-self.num_tasks:].tolist()
        chem_graph_labels = df[labels_name].values.tolist()
        
        index_list = np.arange(len(smiles))
        # read morpholopy feature
        mo = np.load(os.path.join(self.root, "feature_data", "mo.npz"))['features']
        mo = torch.tensor(mo[index_list], dtype=torch.float32)
        
        # read gene expression feature
        ge = np.load(os.path.join(self.root, "feature_data", "ge.npz"))['features']
        ge = torch.tensor(ge[index_list], dtype=torch.float32)
        
        graph = []
        num = len(smiles)
        for i in tqdm(range(num)):
            smile = smiles[i]
            fingerprint = features[i]
            labels = chem_graph_labels[i]
            x, edge_attr, edge_index = smileToList(smile)
            labels = torch.tensor(labels)
            # if all labels of graph are none, ignore it 
            if all(y is torch.nan for y in labels):
                continue
            if all(y == -1 for y in labels):
                continue
            
            g = Data()
            g.num_nodes = len(x)
            g.x, g.edge_attr, g.edge_index = x, edge_attr, edge_index
            g.cs_label = 1
            
            g.fingerprint = fingerprint.unsqueeze(0)
            g.mo = mo[i].unsqueeze(0)
            if torch.isnan(mo[i]).any().item():
                g.mo_label = 0
                g.mo = torch.zeros_like(g.mo)
            else:
                g.mo_label = 1
            
                
            g.ge = ge[i].unsqueeze(0)
            if torch.isnan(ge[i]).any().item():
                g.ge_label = 0
                g.ge = torch.zeros_like(g.ge)
            else:
                g.ge_label = 1
                
            g.y = labels.view(1, -1)    
            graph.append(g)
            
        data, slices = self.collate(graph)
        torch.save((data, slices), self.processed_paths[0])
        
        # split data
        # rondom split
        file_path = os.path.join(os.path.join(self.root,"split", "random"))
        if not os.path.exists(file_path):
            files = ["train", "test", "valid"]
            os.makedirs(file_path)
            split_data = random_split(np.array(range(len(smiles))), chem_graph_labels, valid=True) 
            for f in files:
                with open(os.path.join(file_path, f+".csv"),'w',encoding='utf8',newline='') as fw :
                    writer = csv.writer(fw)
                    writer.writerow(["index"])
                    writer.writerows(split_data[f][:,np.newaxis])
        
        # scaffold split
        file_path = os.path.join(os.path.join(self.root, "split", "scaffold"))
        if not os.path.exists(file_path):
            files = [ "test", "train", "valid"]
            os.makedirs(file_path, exist_ok=True)
            split_data = scaffold_split(smiles_list=smiles, test_ratio=0.2, valid_ratio=0.1) 
            for f in files:
                with open(os.path.join(file_path, f+".csv"),'w',encoding='utf8',newline='') as fw :
                    writer = csv.writer(fw)
                    writer.writerow(["index"])
                    split_data[f].sort()
                    for value in split_data[f]:
                        writer.writerow([value])
            
        # cross validation and scaffold split
        k = 5
        file_path = os.path.join(self.root, "split", "kfold")
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
            split_smiles = smiles
            files = [ "test", "train"]
            for i in range(k):
                split_data = scaffold_split(smiles_list = split_smiles, test_ratio = 1.0 / (k - i)) 
                split_smiles = [smiles[i] for i in split_data['train']]
                split_data['train'] = list(set(range(len(smiles))) - set(split_data['test']))
                f = os.path.join(file_path, str(i))
                os.makedirs(f, exist_ok=True)
                for j in range(2):
                    with open(os.path.join(f, files[j]+".csv"),'w',encoding='utf8',newline='') as fw :
                        writer = csv.writer(fw)
                        writer.writerow(["index"])
                        split_data[files[j]].sort()
                        for value in split_data[files[j]]:
                            writer.writerow([value])
        
        
    def get_idx_split(self, split_type = "kfold"):
        path = os.path.join(self.root, "split", split_type)
        if split_type == "random":
            file = os.listdir(path)
            file.sort()
            split_index = {}
            for f in file:
                df = pd.read_csv(os.path.join(path, f))
                split_index[f.split(".")[0]] = df["index"].tolist()
            return split_index
        elif split_type == "kfold":
            dirs = os.listdir(path)
            dirs.sort()
            split_index = []
            for d in dirs:
                train_test = {}
                files = ['train', 'test']
                for f in files:
                    df = pd.read_csv(os.path.join(path, d, f+'.csv'))
                    train_test[f] = df["index"].tolist()
                split_index.append(train_test)
            return split_index
        elif split_type == 'scaffold':
            file = os.listdir(path)
            file.sort()
            split_index = {}
            for f in file:
                df = pd.read_csv(os.path.join(path, f))
                split_index[f.split(".")[0]] = df["index"].tolist()
            return split_index
        else:
            raise("wrong split type")
        

if __name__ == "__main__":
    root = "./dataset"
    dataset_name = ['chembl2k', "broad6k"]
    num_tasks = [41, 32]
    data_files = [{"assay_data":"assays.csv.gz", 
                    "mo": "CP-JUMP.csv.gz",
                    "mo_feature": "CP-JUMP_feature.npz", 
                    "ge": "GE.csv.gz", 
                    "ge_feature": "GE_feature.npz"}, 
                  
                  {"assay_data":"assays.csv.gz", 
                   "mo": "CP-Bray.csv.gz",
                   "mo_feature": "CP-Bray_feature.npz",
                   "ge": "GE.csv.gz",
                   "ge_feature": "GE_feature.npz"}
                  ]
    
    for i in range(len(num_tasks)):
        print(dataset_name[i], num_tasks[i])
        dataset = mergeDataset(root=root, dataset_name=dataset_name[i], num_tasks=num_tasks[i], data_file=data_files[i])
        print(dataset[0].x.shape, dataset[0].y.shape)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers = 0)
        for batch in train_loader:
            print(batch.y.shape)
            break
        
        # kfold
        split_idx = dataset.get_idx_split(split_type="kfold")
        for i in range(len(split_idx)):
            train_loader = DataLoader(dataset[split_idx[i]['train']], batch_size=32, shuffle=True, num_workers=0)
            test_loader = DataLoader(dataset[split_idx[i]['test']], batch_size=32, shuffle=False, num_workers=0)
            for batch in train_loader:
                print(batch.y.shape)
                break
        
        # scaffold
        split_idx = dataset.get_idx_split(split_type="scaffold")
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False, num_workers=0)
        for batch in train_loader:
            print(batch.y.shape)
            break
        
        # random
        split_idx = dataset.get_idx_split(split_type="random")
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False, num_workers=0)
        train_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False, num_workers=0)
        for batch in train_loader:
            print(batch.y.shape, batch.fingerprint.shape, batch.ge.shape)
            break
    
