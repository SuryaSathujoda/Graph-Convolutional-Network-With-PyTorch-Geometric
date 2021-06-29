import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

class CoraGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(CoraGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.ln = LayerNorm(hidden_dim)

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln(x)

        x = self.conv2(x, edge_index)
        emb = x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin1(x)
        x = F.dropout(x, p=0.75)
        x = self.lin2(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label, mask):
        pred = pred[mask]
        label = label[mask]
        return F.nll_loss(pred, label)

def train(model, data):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        _, pred = model(data)
        loss = model.loss(pred, data.y, data.train_mask)
        loss.backward()
        optimizer.step()

        if(epoch % 10 == 0):
            model.eval()
            _, pred = model(data)
            _, pred = pred.max(dim=1)
            correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            acc = correct / data.val_mask.sum().item()
            print('Epoch: {}. Accuracy: {:,.4f}'.format(epoch+1, acc))

    return model

def test(model, data):
    model.eval()
    _, pred = model(data)
    _, pred = pred.max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print('Test Accuracy: {:,.4f}'.format(acc))


dataset = Planetoid(root='./data/cora/', name='Cora')
data = dataset[0]

model = CoraGCN(1433, 64, 7, 0.75)

model = train(model, data)
model = test(model, data)
