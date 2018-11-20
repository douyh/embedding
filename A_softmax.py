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
    def __init__(self, input_size, output_size,  m = 4, normalization = True, min_lambda = 5, max_lambda = 1500):
        super(AngleLinear, self).__init__()
        self.it = 0
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.cur_lambda = max_lambda
        self.input_size = input_size # D
        self.output_size = output_size # C
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        
        self.m = m
        self.normalization = normalization
        self.multi_theta = [
            lambda x: 1,
            lambda x: x,
            lambda x: 2 * (x ** 2) - 1,
            lambda x: 4 * (x ** 3) - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * (x ** 5) - 20 * (x ** 3) + 5 * x
            ]
    def forward(self, x, target):
        '''
        x: input features (N, D) N is batch size and D is the dimension of features
        '''
        self.it += 1
        
        w = self.weight # (D, C)

        if self.normalization: # Asoftmax else Lsoftmax
           w = F.normalize(w, p = 2, dim = 0)
        w_norm = w.norm(p = 2, dim = 0).unsqueeze(0) #(1, C)
        x_norm = x.norm(p = 2, dim = 1).unsqueeze(1) #(N, 1)
        
        cos_theta = x.mm(w) # (N, C)
        cos_theta = cos_theta / x_norm / w_norm 
        cos_theta = cos_theta.clamp(-1, 1)

        cos_theta_target = cos_theta[range(len(target)), target] * 1.0 # (N, ) 加上这个target似乎可以减少计算
        cos_m_theta = self.multi_theta[self.m](cos_theta_target) # get cos(m * theta) (N, )
        theta = cos_theta_target.acos()
        k = (self.m * theta / 3.141592653).floor() # find k
        n_one = k * 0.0 - 1
        phi_theta_target = (n_one ** k) * cos_m_theta - 2 * k # (N, 1)
        
        self.cur_lambda = max(self.min_lambda, self.max_lambda / (1 + 0.1 * self.it))
        addition = (phi_theta_target - cos_theta_target)  *  x_norm.view(-1) / (1 + self.cur_lambda)
        cos_theta = cos_theta * x_norm
        cos_theta[range(len(target)), target] += addition
        
        return cos_theta


