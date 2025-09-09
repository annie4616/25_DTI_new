import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import pandas as pd


from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import train, DrugProtModel

from model import GCN
from utils import (
    train
    # test,
    # parity_plot,
    # loss_curve,
    # Standardizer,
)
from graphs import DrugProteinDataset, collate

#### Fix seeds
np.random.seed(0)
torch.manual_seed(0)
use_GPU = torch.cuda.is_available()
device = torch.device("cuda:0")
# datasets = ['kiba']
# cuda_name = "cuda:0"

## inputs
max_atoms = 200
node_vec_len = 60
train_size = 0.7
batch_size = 32
hidden_nodes = 60
n_conv_layers = 2
n_hidden_layers = 2
LR = 0.001
n_epochs = 10
protein_embed_path = '/ssd0/sohyun/25_DTI/cache/protein_embeds_esm2_650m_mean.pt'

#### Start by creating dataset
main_path = Path(__file__).resolve().parent
data_path = main_path / "data" / "kiba.csv"
dataset = DrugProteinDataset(
    dataset_path=data_path, embed_path=protein_embed_path, max_atoms=max_atoms, node_vec_len=node_vec_len
) 
# dataset[0][0][0].shape = (200, 60)   

#### Split data into training and test sets
# Get train and test sizes
dataset_indices = np.arange(0, len(dataset), 1)
train_size = int(np.round(train_size * len(dataset)))
test_size = len(dataset) - train_size

# Randomly sample train and test indices
train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

# Create dataoaders
train_sampler = SubsetRandomSampler(train_indices) # 순서를 무작위로 섞음
test_sampler = SubsetRandomSampler(test_indices) # 이거 안 써도됨됨
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate,
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=collate,
)

# # device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
# gcn = GCN(node_vec_len=node_vec_len,
#     node_fea_len=hidden_nodes,
#     hidden_fea_len=hidden_nodes,
#     n_conv=n_conv_layers,
#     n_hidden=n_hidden_layers,
#     n_outputs=1,
#     p_dropout=0.1).to(device) # 하이퍼파라미터 전달
# # loss_fn = nn.MSELoss()

model = DrugProtModel(
    node_vec_len=node_vec_len,
    node_fea_len=hidden_nodes,
    hidden_fea_len=hidden_nodes,  # GCN 풀링 표현 차원으로 사용
    n_conv=n_conv_layers,
    n_hidden=n_hidden_layers,
    n_outputs=1,
    p_dropout=0.1,
    prot_dim=1280              # ESM2 mean 예시
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Loss function
loss_fn = torch.nn.MSELoss()

best_mse = 1000
best_ci = 0
best_epoch = -1

model_file_name = 'model_GCN_kiba.model'
result_file_name = 'result_GCN_kiba.csv'

#### Train the model
loss = []
mae = []
epoch = []
for i in range(n_epochs):
    epoch_loss, epoch_mae = train(
        i,
        model,
        train_loader,
        optimizer,
        loss_fn,
        use_GPU,
        max_atoms,
        node_vec_len,
    )
    loss.append(epoch_loss)
    mae.append(epoch_mae)
    epoch.append(i)


#### Print final results
print(f"Training Loss: {loss[-1]:.2f}")
print(f"Training MAE: {mae[-1]:.2f}")