import torch
import torch.nn as nn
import torch_geometric.nn as gnn


################################################################GCN
class GCNS(nn.Module):
    def __init__(self,in_channels=8,out_channels=1):
        super(GCNS, self).__init__()
        self.g1= gnn.GCNConv(in_channels=in_channels,out_channels=16,bias=False)
        self.bn1 = gnn.BatchNorm(16)
        self.ac1 = nn.ReLU()
        self.g2= gnn.GCNConv(in_channels=16, out_channels=16,bias=False)
        self.bn2 = gnn.BatchNorm(16)
        self.ac2 = nn.ReLU()
        self.g3 = gnn.GCNConv(in_channels=16, out_channels=out_channels)

    def forward(self, x,es):
        x = self.g1(x,es)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.g2(x,es)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.g3(x, es)
        x = torch.sigmoid(x)
        return x


class GCNRes(nn.Module):
    def __init__(self,channels=8,inner_channels=None):
        super(GCNRes, self).__init__()
        if inner_channels is None:
            inner_channels=channels*2
        self.g1= gnn.GCNConv(in_channels=channels,out_channels=inner_channels,bias=False)
        self.bn1=gnn.BatchNorm(inner_channels)
        self.ac1=nn.ReLU()
        self.g2= gnn.GCNConv(in_channels=inner_channels,out_channels=channels,bias=False)
        self.bn2 = gnn.BatchNorm(channels)
        self.ac2 = nn.ReLU()

    def forward(self, x,es):
        res=x
        x = self.g1(x,es)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.g2(x,es)
        x = self.bn2(x)
        x=  self.ac2(x+res)
        return x

class GCND(nn.Module):
    def __init__(self,in_channels=8,out_channels=1):
        super(GCND, self).__init__()
        self.g1 = gnn.GCNConv(in_channels=in_channels, out_channels=16,bias=False)
        self.bn1 = gnn.BatchNorm(16)
        self.ac1 = nn.ReLU()
        self.stage1 = GCNRes(16)
        # self.stage2 = GCNRes(16)
        self.header=gnn.GCNConv(in_channels=16, out_channels=out_channels)
        self.sig=nn.Sigmoid()

    def forward(self, x,es):
        x = self.g1(x,es)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.stage1(x,es)
        # x = self.stage2(x,es)
        x=  self.header(x,es)
        x = self.sig(x)
        return x

class GCND2(nn.Module):
    def __init__(self,in_channels=8,out_channels=1):
        super(GCND2, self).__init__()
        self.g1 = gnn.GCNConv(in_channels=in_channels, out_channels=16,bias=False)
        self.bn1 = gnn.BatchNorm(16)
        self.ac1 = nn.ReLU()
        self.stage1 = GCNRes(16)
        self.stage2 = GCNRes(16)
        self.header=gnn.GCNConv(in_channels=16, out_channels=out_channels)
        self.sig=nn.Sigmoid()

    def forward(self, x,es):
        x = self.g1(x,es)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.stage1(x,es)
        x = self.stage2(x,es)
        x=  self.header(x,es)
        x = self.sig(x)
        return x
################################################################MLP
class GCNSL(nn.Module):
    def __init__(self,in_channels=8,out_channels=1):
        super(GCNSL, self).__init__()
        self.g1= gnn.Linear(in_channels=in_channels,out_channels=15,bias=False)
        self.bn1 = gnn.BatchNorm(15)
        self.ac1 = nn.ReLU()
        self.g2= gnn.Linear(in_channels=15, out_channels=20,bias=False)
        self.bn2 = gnn.BatchNorm(20)
        self.ac2 = nn.ReLU()
        self.g3 = gnn.Linear(in_channels=20, out_channels=out_channels)

    def forward(self, x,es):
        x = self.g1(x)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.g2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.g3(x)
        x = torch.sigmoid(x)
        return x

class GCNResL(nn.Module):
    def __init__(self,channels=8,inner_channels=None):
        super(GCNResL, self).__init__()
        if inner_channels is None:
            inner_channels=channels*2
        self.g1= gnn.Linear(in_channels=channels,out_channels=inner_channels,bias=False)
        self.bn1=gnn.BatchNorm(inner_channels)
        self.ac1=nn.ReLU()
        self.g2= gnn.Linear(in_channels=inner_channels,out_channels=channels,bias=False)
        self.bn2 = gnn.BatchNorm(channels)
        self.ac2 = nn.ReLU()

    def forward(self, x,es):
        res=x
        x = self.g1(x)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.g2(x)
        x = self.bn2(x)
        x=  self.ac2(x+res)
        return x

class GCNDL(nn.Module):
    def __init__(self,in_channels=8,out_channels=1):
        super(GCNDL, self).__init__()
        self.g1 = gnn.Linear(in_channels=in_channels, out_channels=16,bias=False)
        self.bn1 = gnn.BatchNorm(16)
        self.ac1 = nn.ReLU()
        self.stage1 = GCNResL(16)
        self.stage2 = GCNResL(16)
        self.header=gnn.Linear(in_channels=16, out_channels=out_channels)
        self.sig=nn.Sigmoid()

    def forward(self, x,es):
        x = self.g1(x)
        x = self.bn1(x)
        x=  self.ac1(x)
        x = self.stage1(x,es)
        x = self.stage2(x,es)
        x=  self.header(x)
        x = self.sig(x)
        return x

if __name__ == '__main__':

    x=torch.rand(4,8)
    es = torch.tensor([[1, 2, 0], [0, 0, 3]], dtype=torch.long)
    model=GCND()
    y=model(x,es)






