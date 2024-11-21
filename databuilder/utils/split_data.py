from sklearn.model_selection import train_test_split, KFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

def random_split(index_list, y, valid=False, test_rate=0.2):
    x_valid = 0
    x_train, x_test, y_train, _ = train_test_split(index_list, y, test_size=test_rate)
    if valid:
        x_train, x_valid, _, _ = train_test_split(x_train, y_train, test_size=0.1)
    
    return {"train": x_train, "test": x_test, "valid": x_valid}

def k_flod(index_list, k):
    kf = KFold(n_splits = int(k), shuffle=True)
    return kf.split(index_list)

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)

def scaffold_split(smiles_list, test_ratio=0.1, valid_ratio = 0, include_chirality=False):
    scaffold_to_smiles = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality)
        scaffold_to_smiles[scaffold].append(i)
    
    scaffolds = list(scaffold_to_smiles.keys())
    random.shuffle(scaffolds)
    
    # num_test = int(len(smiles_list) * test_ratio)
    test_cutoff = int(test_ratio * len(smiles_list))
    valid_cutoff = int(valid_ratio * len(smiles_list))
    
    train_smiles_idx = []
    test_smiles_idx = []
    valid_smiles_idx = []
    
    
    for scaffold in scaffolds:
        if len(test_smiles_idx) < test_cutoff:
            test_smiles_idx.extend(scaffold_to_smiles[scaffold])
        elif len(valid_smiles_idx) < valid_cutoff:
            valid_smiles_idx.extend(scaffold_to_smiles[scaffold])
        else:
            train_smiles_idx.extend(scaffold_to_smiles[scaffold])
            
    
    return {"train": train_smiles_idx,  "test": test_smiles_idx, "valid": valid_smiles_idx}