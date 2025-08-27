import os
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
from torch.utils.data import Dataset, dataloader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Union
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles
from torch_geometric import data as Data # 주로 상수를 이렇게 대문자 온리로 표현
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import GCNConv, global_add_pool
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

# 단백질 임베딩
def load_protein_embeddings(
        pt_path: Union[str, Path],
        protein_ids: List[str]=None,
        dtype=torch.float32):
    obj = torch.load(pt_path, map_location='cpu')

     # case A: 이미 dict 형식
    if isinstance(obj, dict):
        # 텐서형 값만 dtype 맞춰 통일
        out = {}
        for k, v in obj.items():
            tv = torch.as_tensor(v, dtype=dtype)
            if tv.dim() == 2 and tv.size(0) == 1:  # [1, D]로 저장된 경우
                tv = tv.squeeze(0)
            out[k] = tv
        return out

    # case B: (Tensor, List[str]) 튜플
    if isinstance(obj, tuple) and len(obj) == 2:
        embeds, ids = obj
        embeds = embeds.to(dtype=dtype)
        return {pid: embeds[i] for i, pid in enumerate(ids)}

    # case C: Tensor만 있고 id 목록은 외부로 받음
    if torch.is_tensor(obj):
        assert protein_ids is not None, \
            "Tensor만 저장돼 있으면 protein_ids 리스트를 같이 전달해야 합니다."
        embeds = obj.to(dtype=dtype)
        assert len(protein_ids) == embeds.size(0), \
            "protein_ids 길이와 임베딩 N이 일치해야 합니다."
        return {pid: embeds[i] for i, pid in enumerate(protein_ids)}

    raise ValueError("알 수 없는 저장 포맷입니다. dict / (Tensor, ids) / Tensor 중 하나여야 합니다.")

# GCN 모델 구현
class GCN(torch.nn.Module):
    def __init__(self, n_output = 1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt = 25, output_dim=128, dropout=0.2, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaasm_in_dim=1280):
        super(GCN, self).__init__()
        self.n_output = n_output # 모델의 출력은 숫자 1개

        # Drug Representation
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd*4)

        # fully connected layer - 1024차원으로 변환
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)

        # activation function
        self.relu = nn.ReLU()
        #Dropout
        self.dropout = nn.Dropout(dropout)

        # Drug + Protein Representation fusion
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024,512)

        self.out = nn.Linear(512, self.n_output)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # target = data.target

        # GCN Layer
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = gmp(x, batch)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # Drug, Protein Representation을 torch.cat을 이용해 하나로 결합
        # xc = torch.cat((x, xt), 1)
        x = xc

        # 하나의 출력값 구하기
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
    
# 데이터셋 클래스
class DrugProteinDataset(Dataset): # 원래 많이 쓰는 데이터셋 사용하기
    def __init__(self, xd, xt, y, smile_graph, protein_embed_path='./cache/protein_embeds_esm2_650m_mean.pt', esm_model=None, vocab=None, strict=True):
        assert len(xd) == len(xt) == len(y), 'xd,y의 길이는 같아야 합니다'
        self.xd = xd
        self.xt = xt
        self.y = y
        self.smile_graph = smile_graph
        self.esm_model = esm_model
        self.vocab = vocab
        self.strict = strict

        self.protein_embed_dict = torch.load(protein_embed_path, map_location="cpu")

        for k,v in self.protein_embed_dict.items():
            if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                self.protein_embed_dict[k] = v.float
        self.protein_embeds = []
        for t in self.xt:
            emb = self.protein_embed_dict.get(t, None)
            self.protein_embeds.append(emb)
    def __len__(self):
        return len(self.xd)
    def __getitem__(self, idx):
        d = self.xd[idx]
        t = self.xt[idx]
        y = self.y[idx]

        c_size, features, edge_index = smiles_to_graph(d)
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor([y], dtype=torch.float32)
        )
        data.c_size = torch.tensor([c_size])

        # protein ESM (pooled vector)
        data.protein = torch.tensor(self.esm[t], dtype=torch.float32)  # shape: [esm_dim]

        return data

def pyg_collate(batch):
    # PYG data들을 batch로 묶는 collate 함수
    # 배치는 반드시 collate_fn = Batch.from_data_list로 지정해야 함
    return Batch.from_data_list(batch)

# train/test
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, data.y.view(-1,1).float().to(device))

        loss.backward()
        optimizer.step()

        if batch_idx%LOG_INTERVAL == 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)]|tLoss: {:.6f}'.format(epoch,
                                                                          batch_idx*len(data.x),
                                                                          len(train_loader.dataset),
                                                                          100.*batch_idx/len(train_loader),
                                                                          loss.item()))
            
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = ['kiba']
modelling = ['GCN']
cuda_name = "cuda:0"

Train_Batch_Size = 512
Test_Batch_Size = 512
LR = 0.0005
Log_Interval = 20
Num_Epochs = 10

# 1) 임베딩 로드
esm_dict = load_protein_embeddings(
    "./cache/protein_embeds_esm2_650m_mean.pt",
    # protein_ids=...  # 텐서 단독 저장 케이스일 때만 필요
    dtype=torch.float32
)
train_d = pd.read_csv('./data/kiba_train.csv')
test_d = pd.read_csv('./data/kiba_test.csv')

#     def __init__(self, xd, xt, y, smile_graph, protein_embed_path='protein_embeds_esm2_650m_mean.pt', esm_model=None, vocab=None, strict=True):

train_data = DrugProteinDataset(train_d['X1'], train_d['X2'], train_d['Y'], smile_graph)
test_data = DrugProteinDataset(test_d['X1'], test_d['X2'], test_d['Y'], smile_graph)

train_loader = DataLoader(train_data, batch_size=Train_Batch_Size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=Test_Batch_Size, shuffle=False)

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
model = modelling[0]().to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_mse = 1000
best_ci = 0
best_epoch = -1

model_file_name = 'model_GCN_kiba.model'
result_file_name = 'result_GCN_kiba.csv'

for epoch in range(Num_Epochs):
    train(model, device, train_loader, optimizer, epoch + 1)

    G, P = predicting(model, device, test_loader)

    ret = [rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P)]

    if ret[1] < best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name, 'w') as f:
            f.write(','.join(map(str,ret)))
        best_mse = ret[1]
        best_ci = ret[-1]
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,modelling[0],datasets[0])
    else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,modelling[0],datasets[0])