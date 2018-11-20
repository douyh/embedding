import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import numpy as np
class AdaptiveNormalize(nn.Module):
    def __init__(self, scale):
        super(AdaptiveNormalize, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([scale]))

    def forward(self, x):
        x = F.normalize(x, p = 2, dim = 1)
        x = x * self.scale
        return x
class fast_pair_loss(nn.Module):
    def __init__(self):
        super(fast_pair_loss, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size = 10, stride = 10)
        self.index = torch.zeros([10, 10, 2])
    def forward(self, x, target, hard_ratio = 1,  margin = 2, factor = 10):
        '''
        x : input N D
        target: label N
        '''
        # calculate the distance of all the vectors
        norm_raw = x.norm(p = 2, dim = 1)
        norm_col = norm_raw.unsqueeze(1)
        dist_matrix = norm_raw.pow(2) + norm_col.pow(2) - 2 * x.mm(x.t()) # N N

        # use a matrix to discriminate postive pairs and negtive pairs
        batch_size = len(target)
        pair_matrix = torch.eye(batch_size).cuda()
        for i in range(10):
            pair_matrix[i * 10: i * 10 + 10, i * 10: i * 10 + 10] += 1
        
        # calculate the loss
        loss_matrix = dist_matrix
        loss_matrix[pair_matrix == 0] = margin - loss_matrix[pair_matrix == 0]
        loss_matrix[loss_matrix < 0] = 0
        pos_loss = loss_matrix[pair_matrix == 1]
        neg_loss = loss_matrix[pair_matrix == 0]
        
        # sort loss and choose highest
        hard_pos_cnt = int(len(pos_loss) * hard_ratio)
        hard_neg_cnt = int(len(neg_loss) * hard_ratio)
        hard_pos_loss = torch.sort(pos_loss, descending = True)[0][0:hard_pos_cnt]
        hard_neg_loss = torch.sort(neg_loss, descending = True)[0][0:hard_neg_cnt]
        hard_pos_loss = hard_pos_loss[hard_pos_loss > 0]
        hard_neg_loss = hard_neg_loss[hard_neg_loss > 0]
        
        # get finnal loss
        hard_pos_cnt, hard_neg_cnt = len(hard_pos_loss), len(hard_neg_loss)
        hard_loss = hard_pos_loss.sum() + hard_neg_loss.sum() * factor
        hard_loss = hard_loss / (hard_pos_cnt + hard_neg_cnt)

        # get the index matrix
        for i in range(0, batch_size, 10):
            s1 = target[i]
            for j in range(0, batch_size, 10):
                s2 = target[j]
                self.index[i // 10][j // 10][0] = s1
                self.index[i // 10][j // 10][1] = s2
       
        # get average loss of two categories
        avg_loss = loss_matrix.unsqueeze(0)
        avg_loss = self.avg_pool(avg_loss)
        avg_loss = avg_loss.squeeze(0)
        return hard_loss, self.index, avg_loss

