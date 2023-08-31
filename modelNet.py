# -*- coding: utf-8 -*-
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        D_in, H, D_out = 2, 30, 1
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.output = nn.Linear(H, D_out)
        self.relu = nn.ReLU()
        self.activate = nn.Sigmoid()


    def forward(self,x):
        res = self.relu(self.linear1(x))
        res = self.relu(self.linear2(res))
        res = self.output(res)
        res = self.activate(res)
        return res

