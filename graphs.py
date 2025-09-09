import os
# GPU 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# cpu, gpu 지원하는 과정이 달라서 전처리는 대부분 cpu해서 저장해놓고 돌릴 때 gpu로 모델, 데이터 올려서 돌림.
# cpu에서 할 때는 np.array형태, gpu 올릴 때는 torch.tensor 형태로 바꿔서 올림.
# 데이터셋에서 텐서로 묶어주는 것이 좋음
import sys
import torch
import os
import numpy as np
import pandas as pd
import networkx as nx
from math import sqrt
from random import shuffle
from IPython.display import SVG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem import MolFromSmiles
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset
print('executed')

class Graph:
    def __init__(self, molecule_smiles: str, node_vec_len: int, max_atoms: int = None):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()
    def smiles_to_mol(self):
        # Use MolFromSmiles from RDKit to get molecule object
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            return

        # Add hydrogens to molecule
        self.mol = Chem.AddHs(mol)

    def smiles_to_graph(self):
        """
        Converts smiles to a graph.
        """

        # Get list of atoms in molecule
        atoms = self.mol.GetAtoms()

        # Create empty node matrix
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        # Iterate over atoms and add to node matrix
        for atom in atoms:
            # Get atom index and atomic number
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()

            # Assign to node matrix
            node_mat[atom_index, atom_no] = 1

        # Create empty adjacency matrix
        adj_mat = np.zeros((n_atoms, n_atoms))

        # Create adjacency matrix
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)

        # Create distance matrix
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1

        # Get modified adjacency matrix
        adj_mat = adj_mat * (1 / dist_mat)

        # Pad the adjacency matrix
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(
            adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant"
        )

        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        # adj_mat = adj_mat + np.eye(n_atoms)

        # Save both matrices
        self.node_mat = node_mat
        self.adj_mat = adj_mat
class ProteinEmbedding:
    _embeds = None  # 클래스 캐시
    @classmethod
    def load(cls, path): # 이 클래스 안에서 임베딩을 한번만 불러와서 저장해둘 수 있음
        if cls._embeds is None:
            cls._embeds = torch.load(path)
    def __init__(self, protein_id: str,protein_seq: str):
        self.id = protein_id
        self.seq = protein_seq
        # self.embeddings = torch.load(protein_embed_path, map_location="cpu")
        # self.get_esm_embedding()
    def get(self):
        return self._embeds[self.id]
    def get_esm_embedding(protein_id: str, path: str):
        torch.save({protein_id: ProteinEmbedding._embeds[protein_id]}, path)



# 데이터셋 클래스
class DrugProteinDataset(Dataset): # 원래 많이 쓰는 데이터셋 사용하기
    def __init__(self, dataset_path:str, embed_path: str, node_vec_len: int, max_atoms: int):
        # assert len(xd) == len(xt) == len(y), 'xd,y의 길이는 같아야 합니다'
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        df = pd.read_csv(dataset_path)

        self.indices = df.index.to_list()
        self.smiles = df["X1"].to_list()
        self.protein = df["X2"].to_list()
        self.prot_id = df["ID2"].to_list()
        self.outputs = df["Y"].to_list()
        self.embed_path = embed_path
        ProteinEmbedding.load(self.embed_path)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i: int):
        smile = self.smiles[i]

        # Create MolGraph object
        mol = Graph(smile, self.node_vec_len, self.max_atoms)
        # Get matrices
        node_mat = torch.Tensor(mol.node_mat)
        adj_mat = torch.Tensor(mol.adj_mat)

        output = torch.Tensor([self.outputs[i]])

        seq = self.protein[i]
        pid = self.prot_id[i]
        emb = ProteinEmbedding(pid, seq).get()

        return (node_mat, adj_mat), output, emb
        
def collate(dataset: Dataset):
    node_mats = []
    adj_mats = []
    outputs = []
    embs = []
    # smiles = []

    for i in range(len(dataset)):
        (node_mat, adj_mat), output, emb = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        embs.append(emb)
        # smiles.append(smile)

    node_mats_tensor = torch.cat(node_mats, dim = 0)
    adj_mats_tensor = torch.cat(adj_mats, dim=0)
    outputs_tensor = torch.cat(outputs, dim=0)
    embs_tensor = torch.cat(embs, dim=0)

    return (node_mats_tensor, adj_mats_tensor), embs_tensor, outputs_tensor

if __name__ == "__main__":
    dataset_path = '/ssd0/sohyun/25_DTI/data/kiba.csv'
    data = DrugProteinDataset(dataset_path, '/ssd0/sohyun/25_DTI/cache/protein_embeds_esm2_650m_mean.pt',20, 75)
    print(torch.diag(data[0][0][1].sum(dim=-1)))
    # emb size = 1280
    # print(collate(data)[0][1])