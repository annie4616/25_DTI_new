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
import json,pickle
import networkx as nx
from math import sqrt
from random import shuffle
from collections import OrderedDict
from scipy import stats
from IPython.display import SVG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Union
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data, Batch # 주로 상수를 이렇게 대문자 온리로 표현
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

# 시각화 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline
print('executed')

# 약물 인코딩
def feature_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input{0} not allowed in set {1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
def feature_encoding_unk(x, allowable_set):
    # allowable set에 있지 않으면 마지막 요소로 매핑
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_features(atom):
    return np.array(feature_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    feature_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
                    feature_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4,5,6,7,8,9,10]) +
                    feature_encoding_unk(atom.GetImplicitValence(),[0,1,2,3,4,5,6,7,8,9,10]) +
                    [atom.GetIsAromatic()] # 이거 하나하나 어떻게 나오는건지 다음에 공부해보기
                    )

# returns: 원자 개수, 원자 특성 행렬, 인접 행렬
def smiles_to_graph(smiles):
    # 문자열 -> 그래프
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 원자 개수 저장
    c_size = mol.GetNumAtoms()
    features = []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/sum(feature)) # 정규화 - feature들이 모두 숫자로 표현될 수 있는건가?

    # 엣지 - 시작 원자 정보와 끝 원자 정보
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # nx라이브러리를 이용해 데이터를 방향 그래프로 변환
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index
