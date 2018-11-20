import torch
import torch.nn as nn
import torch.nn.functional as F
class AdaptiveNormalize(nn.Module):
    def __init__(self):
        super(AdaptiveNormalize, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([10]))

    def forward(self, x):
        x = F.normalize(x, p = 2, dim = 1)
        x = x * self.scale
        return x

class AngleLinear(nn.Module):
    def __init__(self, input_size, output_size, s = 30, m = 0.35):
        super(AngleLinear, self).__init__()
        self.input_size = input_size # D
        self.output_size = output_size # C
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.s = s
        self.m = m
    def forward(self, x, target):
        '''
        x: input features (N, D) N is batch size and D is the dimension of features
        '''
        
        w = self.weight # (D, C)
        w = F.normalize(w, p = 2, dim = 0)
        w_norm = w.norm(p = 2, dim = 0).unsqueeze(0) #(1, C)
        x_norm = x.norm(p = 2, dim = 1).unsqueeze(1) #(N, 1)
        
        cos_theta = x.mm(w) # (N, C)
        cos_theta = cos_theta / x_norm / w_norm 
        cos_theta = cos_theta.clamp(-1, 1)
        
        cos_theta[range(len(target)), target] -= self.m
        cos_theta = cos_theta * self.s
        
        return cos_theta


