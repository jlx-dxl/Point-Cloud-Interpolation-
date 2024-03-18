import torch
import torch.nn as nn
import torch.nn.functional as F


class Curve_Fitting(nn.Module):
    def __init__(self, field, degree):
        super(Curve_Fitting, self).__init__()
        self.conv = nn.Conv1d(2*field+1, 128, 1, bias=True)



    def forward(self, data):
        '''
        :param data: [B,#number,N]   #number=2*field+1
        :return coefficients: [B,#power,N]   #power<=2*field,>0
        '''



        coefficients = data
        return coefficients
