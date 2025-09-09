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
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

# 시각화 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline
print('executed')


class PoolingLayer(nn.Module): # mean pooling
    """
    Create a pooling layer to average node-level properties into graph-level properties
    """

    def __init__(self):
        # Call constructor of base class
        super().__init__()

    def forward(self, node_fea):
        # Pool the node matrix
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea

# GCN 모델 구현
class GCN(torch.nn.Module):
    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        # self.conv1 = GCNConv(num_features_xd, 128) # 보통 2의 제곱수로 맞춤
                # Define layers
        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        # Convolution layers
        self.conv_layers = nn.ModuleList()
        in_dim = node_fea_len
        sizes = [in_dim, 128]
        for i in range(n_conv):
                self.conv_layers.append(GCNConv(sizes[i], sizes[i+1]))
                sizes.append(sizes[i+1]*2)
        final_gcn_dim = sizes[-2]

        # Pool convolution outputs
        self.pooling = PoolingLayer()
        pooled_node_fea_len = final_gcn_dim

        # Pooling activation
        self.pooling_activation = nn.LeakyReLU()

        # From pooled vector to hidden layers
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)

        # Hidden layer activation function
        self.hidden_activation = nn.LeakyReLU()

        # Hidden layer dropout
        self.dropout = nn.Dropout(p=p_dropout)

        # If hidden layers more than 1, add more hidden layers
        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList(
                [self.hidden_layer for _ in range(n_hidden - 1)]
            )
            self.hidden_activation_layers = nn.ModuleList(
                [self.hidden_activation for _ in range(n_hidden - 1)]
            )
            self.hidden_dropout_layers = nn.ModuleList(
                [self.dropout for _ in range(n_hidden - 1)]
            )

        # Final layer going to the output
        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)
        
    def forward(self, node_mat, adj_mat):
        # Perform initial transform on node_mat
        # node_fea = self.init_transform(node_mat)

        # Perform convolutions
        # for conv in self.conv_layers:
        #     node_fea = conv(node_fea, adj_mat)
        # x = node_mat.squeeze(0)     # (N,F)
        # adj = adj_mat.squeeze(0)    # (N,N)
        B, N, F = node_mat.shape
        x = self.init_transform(node_mat.reshape(B*N, F))  # (B*N, F')
        edge_index, edge_weight = dense_to_sparse(adj_mat)
        edge_index = edge_index.to(x.device).long()

        # x = self.init_transform(x)
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index, edge_weight).relu()
        x = self.conv_layers[-1](x, edge_index, edge_weight)


        # Perform pooling
        # pooled_node_fea = self.pooling(node_fea)
        pooled = x.mean(dim=0, keepdim=True)  # 단일 그래프 평균 풀링
        pooled_node_fea = self.pooling_activation(pooled)

        # First hidden layer
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        # Subsequent hidden layers
        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)

        # Output
        out = self.hidden_to_output(hidden_node_fea)

        return out
    
if __name__ == "__main__":
    from graphs import Graph, ProteinEmbedding

    g = Graph("CC", node_vec_len=20)
    n = torch.Tensor(g.node_mat).view(1, g.node_mat.shape[0], g.node_mat.shape[1])
    a = torch.Tensor(g.adj_mat).view(1, g.adj_mat.shape[0], g.adj_mat.shape[1])

        # 2) 단백질 임베딩 준비
    # ProteinEmbedding.load("/ssd0/sohyun/25_DTI/cache/protein_embeds_esm2_650m_mean.pt")  # 최초 1회만
    # prot = ProteinEmbedding("O00141","MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL" )  # ProteinID와 서열
    # emb = prot.get()
    # prot_emb = emb.unsqueeze(0)  # 배치 차원 추가 → (1, dim)
    model = GCN(
        node_vec_len=20,
        node_fea_len=20,
        hidden_fea_len=10,
        n_conv=2,
        n_hidden=2,
        n_outputs=1,
    )

    with torch.no_grad():
        out = model(n, a)
        print(out)

