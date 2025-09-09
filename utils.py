import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
from model import GCN

#### Fix seeds
np.random.seed(0)
torch.manual_seed(0)
use_GPU = torch.cuda.is_available()

class DrugProtModel(torch.nn.Module):
    def __init__(self,
                 node_vec_len,
                 node_fea_len,
                 hidden_fea_len,
                 n_conv,
                 n_hidden,
                 n_outputs,
                 p_dropout,
                 prot_dim):
        super().__init__()
        self.gcn = GCN(node_vec_len=node_vec_len,
                       node_fea_len=node_fea_len,
                       hidden_fea_len=hidden_fea_len,
                       n_conv=n_conv,
                       n_hidden=n_hidden,
                       n_outputs=n_outputs,
                       p_dropout=p_dropout)
        self.buf = None
        self.gcn.hidden_to_output.register_forward_hook(lambda m, i, o: setattr(self, "buf", o))
        self.head = nn.Sequential(
            nn.Linear(hidden_fea_len + prot_dim, hidden_fea_len),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_fea_len, n_outputs)
        )

    def forward(self, node_mat, adj_mat, prot_emb):
        _ = self.gcn(node_mat, adj_mat)          # GCN은 내부 그대로 사용 - forward hook이 있으므로 buf에 GCN 출력이 저장됨.
        prot_emb = torch.tensor(prot_emb, device=self.buf.device).view(self.buf.size(0), -1)
        z = torch.cat([self.buf, prot_emb], 1)   # (B, hidden_fea_len + prot_dim)
        return self.head(z)

# train/test
def train(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    avg_loss = 0
    avg_mae = 0
    count = 0

    model.train()
    
    for i, dataset in enumerate(training_dataloader):
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]
        prot_emb = dataset[2]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len)) # node_met안의 원소 개수/분자 하나 행렬의 원소 개수 = 총 분자 수
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)


        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda(), prot_emb.cuda())
            nn_output = output.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output

        # Compute output from network
        nn_prediction = model(*nn_input) # 이건 GCN이라 단백질 넣을 수 없음

        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss

        # Calculate MAE
        # prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, nn_prediction)
        avg_mae += mae

        # Set zero gradients for all tensors
        optimizer.zero_grad()

        # Do backward prop
        loss.backward()

        # Update optimizer parameters
        optimizer.step()

        # Increase count
        count += 1
        
            
def predicting(model, device, loader):
    model.eval()
    total_preds = []
    total_labels = [] # torch.tensor() 해야 하나?

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()