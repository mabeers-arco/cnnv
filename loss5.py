"""
Loss 5 assumes symmetry and planarity, and optimizes over a single parameter, slant angle of the symmetry
plane. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from optimize import loss5_torch
from preprocess import get_edge_matrix, get_M_xcol2, get_adj
import datetime
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pickle
import math
from numerical_summary import sos




class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.type, self.weight.type)
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        # print(input.type(), self.weight.type())
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'





class GCN(nn.Module):
    def __init__(self, p, network_size, nonlinearity):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(2, 100)
        self.gc2 = GraphConvolution(100, 1000)
        self.gc3 = GraphConvolution(1000, 100)
        self.linear1 = nn.Linear(700, 100)
        self.linear2 = nn.Linear(100, 1)
        self.drop_out = nn.Dropout(p)
        self.ns = network_size
        self.nonlinearity = nonlinearity
        self.M = get_edge_matrix(network_size)
        self.M_xcol2 = get_M_xcol2(network_size)
        self.device = "cpu"


    def forward(self, xy, adj, mps, extract_xyz = False):
        #b = xy.shape[0]

        z = self.nonlinearity(self.gc1(xy, adj))
        z = self.drop_out(z)
        z = self.nonlinearity(self.gc2(z, adj))
        z = self.drop_out(z)
        z = self.nonlinearity(self.gc3(z, adj)) 
        z = self.nonlinearity(self.linear1(z.view(-1, 700)))
        theta = np.pi/2 * torch.sigmoid(self.linear2(z))


        if extract_xyz: 
            dct = loss5_torch(theta, xy, mps, get_xyz=True)
            return dct
        else:
             return loss5_torch(theta, xy, mps, get_xyz=False)



def train_step(xy, adj, mps, net, opt_fn, loss_fn):
    netout = net(xy, adj, mps)
    #loss = loss_fn(sda_est, sda)
    loss = loss_fn(netout, torch.zeros(len(netout)))
    loss.backward()
    opt_fn.step()
    opt_fn.zero_grad()
    return loss




###################################################################################
# Open Dataset
with open('./data/everything.pickle', 'rb') as handle:
    everything = pickle.load(handle)

k = 4
xy = torch.tensor(everything['xyz_rotated'][k][:, :2], dtype = torch.float)
mpl = everything['mpl'][k]
adj = torch.eye(7)#everything['adj7'][k]


###################################################################################
# Instantiate network
net = GCN(p=.8, network_size=7, nonlinearity=torch.relu)
print("Network Output = ", net(xy, adj, mpl))


###################################################################################
# Define Loss and Optimizer

L1_loss = torch.nn.L1Loss()
L2_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay = 5e-4)

###################################################################################
# Train Network

# start_time = time.time()
# epochs = 1
# loss_hist_train = []
# loss_validate = []
# for epoch in range(epochs):
#     t = time.time()
#     net.train()
#     for xy, adj, triple_mask, face_mask, sym_mask, ps, num_vv, z in iter(train_loader):   
#         loss = train_step(xy, adj, sym_mask, face_mask, ps, net, optimizer, L2_loss)
#         loss_hist_train.append(loss.item())

#     xy, adj, triple_mask, face_mask, sym_mask, ps, num_vv, z = next(iter(test_loader))  
#     net.eval()
#     new_xyz, symmetry, planarity, validate_loss = net(xy, adj, sym_mask, face_mask, ps, extract_xyz=True) 

#     print("epoch {}: Train Loss = {}, Test Loss = {}, Time/epoch = {}".format(epoch, 
#                                                         np.round(loss.item(), 5), 
#                                                         torch.mean(validate_loss),
#                                                         datetime.timedelta(seconds=round(time.time() - t))))
    
    
# print("Elapsed Time = ", datetime.timedelta(seconds=round(time.time() - start_time)))
n_epochs = 3
loss_hist_train = []
net.train()
for epoch in range(n_epochs):
    for k in range(len(everything['uid'])):  
        xyz = torch.tensor(everything['xyz_rotated'][k], dtype = torch.float)
        mpl = everything['mpl'][k]
        trueC = everything['loss5results'][k]
        if len(mpl) < 3 or xyz.shape != (7,3) or type(trueC) is float:
            continue
        xy = xyz[:, :2] 
        adj = get_adj(everything['pairs'][k], 7)
        compactness = net(xy, adj, mpl)
        trueC = torch.tensor(trueC['fun'], dtype = torch.float)
        loss = L2_loss(compactness, trueC)
        loss.backward()
        #compactness.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist_train += [compactness.item()]
        if k % 100 == 0:
            print("Iteration {} Done".format(k))





import matplotlib.pyplot as plt
plt.plot(loss_hist_train)
plt.show()



k = 4
mpl = everything["mpl"][k]
sym = everything["matched_points_less"][k]
triples = everything["triples_less"][k]
faces = everything['faces'][k]
dct = net(xy, adj, mpl, extract_xyz=True)
sos(dct['xyz'].detach().numpy(), sym, faces, triples)




