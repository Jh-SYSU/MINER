from rdkit import Chem
import rdkit
import torch

map_atoms = {"atomicNum": list(range(1, 119)) + ['misc'], 
            "chirality": [
                'CHI_UNSPECIFIED',
                'CHI_TETRAHEDRAL_CW',
                'CHI_TETRAHEDRAL_CCW',
                'CHI_OTHER',
                'misc'
            ], 
            "degree": list(range(0, 11)) + ['misc'],
            "formalCharge" : list(range(-5, 6)) + ['misc'], 
            "numHs": list(range(0, 9)) + ['misc'], 
            "numRadicalElectrons" : list(range(0, 5)) + ['misc'],
            "hybridization" : [
                'SP',
                'SP2',
                'SP3',
                'SP3D',
                'SP3D2',
                'misc',
            ], 
            "isAromatic" : [False, True], 
            "isInRing" : [False, True]
            }

map_bond = {
    "bondType" : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    "stereo" : [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    "isConjugated" : [False, True]
}

def myIndex(a : list, b) -> int :
    """
    Find the first occurrence of an element in the list and return the index.
    Otherwise, return the end of index
    Args:
        a (list): a list 
        b (_type_): a element you want to find 

    Returns:
        int: the index is the position that b first appear in the a
    """
    try:
        return a.index(b)
    except:
        return len(a) - 1
         
def getAtomFeature(atom: rdkit.Chem.rdchem.Atom) -> list[int]:
    """
    extrat feature from atom
    Args:
        atom: a atom of molecule
    Returns:
        list[int]: the list storing the features extrated from atom
    """
    atom_feature = []
    atom_feature.append(myIndex(map_atoms["atomicNum"], atom.GetAtomicNum()))
    atom_feature.append(myIndex(map_atoms["chirality"], str(atom.GetChiralTag())))
    atom_feature.append(myIndex(map_atoms["degree"], atom.GetTotalDegree()))
    atom_feature.append(myIndex(map_atoms["formalCharge"], atom.GetFormalCharge()))
    atom_feature.append(myIndex(map_atoms["numHs"], atom.GetTotalNumHs()))
    atom_feature.append(myIndex(map_atoms["numRadicalElectrons"], atom.GetNumRadicalElectrons()))
    atom_feature.append(myIndex(map_atoms["hybridization"], str(atom.GetHybridization())))
    atom_feature.append(map_atoms['isAromatic'].index(atom.GetIsAromatic()))
    atom_feature.append(map_atoms["isInRing"].index(atom.IsInRing()))

    return atom_feature


def getBondFeature(bond: rdkit.Chem.rdchem.Bond) -> list[int]:
    """
    extrat feature from chemical bond
    Args:
        bond (rdkit.Chem.rdchem.Bond): a chemical bond of molecule

    Returns:
        list[int]: the list storing the features extrated from chemical bond
    """
    bond_feature = []
    bond_feature.append(map_bond["bondType"].index(str(bond.GetBondType())))
    bond_feature.append(map_bond["stereo"].index(str(bond.GetStereo())))
    bond_feature.append(map_bond["isConjugated"].index(bond.GetIsConjugated()))
    
    return bond_feature

def smileToList(smile: str) -> (torch.tensor, torch.tensor, torch.tensor):
    """
    extrat feature from molecule 
    Args:
        smile (str): a molecule descriped by smiles string

    Returns:
        (torch.tensor, torch.tensor, torch.tensor): the first one stores the feature of atoms, the second one stores the feature of chemical feature, 
        the last one stores the two atoms of each bonds.
    """
    mol = Chem.MolFromSmiles(smile)
    # extra atoms' feature
    x = []
    for atom in mol.GetAtoms():
        x.append(getAtomFeature(atom))
    
    
    # extra bonds' feature
    edge_attr = []
    edge_index = [[], []]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        bond_feature = getBondFeature(bond)
        edge_attr += [bond_feature, bond_feature]
        
    # convert to torch.tensor
    x = torch.tensor(x)    
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(edge_index)
        
    return x, edge_attr, edge_index
        
        
if __name__ == "__main__":
    print(smileToList("CC(Nc1nc(nc2ccccc12)N1CCCC1)c1ccccc1")) 