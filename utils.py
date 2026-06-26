import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import random
import torch
from PIL import Image, ImageOps, ImageFilter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from torch.nn.modules.loss import _Loss
from math import exp
from torch.optim.lr_scheduler import _LRScheduler

# poly
class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs] 
    
    
class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        loss = (- targets * log_probs).mean(0).sum()
        return loss     
    
#  NtXent_loss
class NtXent_loss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.batch_size = args.batchSz
        self.nViews = 2
        self.celoss = nn.CrossEntropyLoss()
        self.temperature = args.temperature
        
    def forward(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.nViews)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(0)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(0)
        mask = torch.eye(labels.shape[0], dtype=torch.uint8).cuda(0).bool()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(0)

        logits = logits / self.temperature
        
        final_loss = self.celoss(logits, labels)
        return final_loss  

    #  SRC 中的loss

class softmax_mse_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_logits, target_logits):
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        mse_loss = (input_softmax-target_softmax)**2 
        return torch.sum(mse_loss)
    
class relation_mse_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()

        activations = torch.reshape(activations, (activations.shape[0], -1))
        ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

        similarity = activations.mm(activations.t())
        norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
        norm_similarity = similarity / norm

        ema_similarity = ema_activations.mm(ema_activations.t())
        ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
        ema_norm_similarity = ema_similarity / ema_norm

        similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
        return torch.sum(similarity_mse_loss)


    
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img    

# SLA LOSS

class DistillKL(nn.Module):
    def __init__(self, args):
        super(DistillKL, self).__init__()
        self.T = args.temperature

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
    
def sla(args):
    return SLALoss(args)

def rotation(images):
    size = images.shape[1:]
    return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)


class SLALoss(nn.Module):
    def __init__(self, args):
        super(SLALoss, self).__init__()
        self.args = args
        self.kd = DistillKL(args)
        self.ce = nn.CrossEntropyLoss()
        self.aux_lamb = args.aux_lamb
        self.fc1 = nn.Linear(args.network_channels[-1], 4 * args.num_classes)
        self.fc2 = nn.Linear(args.network_channels[-1] * args.width, 4 * args.num_classes)

    def forward(self, model, bifpn, inputs, targets):
        bs = inputs.size(0)
        rot_inputs = rotation(inputs)

        feats, outputs = model(inputs)
        feats = feats[-self.args.num_features:]
        bi_feats, bi_outputs = bifpn(feats, self.args.distill == 'od')

        rot_feats, _ = model(rot_inputs)
        rot_feats = rot_feats[-self.args.num_features:]
        rot_bi_feats, _ = bifpn(rot_feats, self.args.distill == 'od')

        last_feat = F.adaptive_avg_pool2d(rot_feats[-1], (1, 1)).view(4 * bs, -1)
        aux_outputs = self.fc1(last_feat)
        rot_last_feat = F.adaptive_avg_pool2d(rot_bi_feats[-1], (1, 1)).view(4 * bs, -1)
        biaux_outputs = self.fc2(rot_last_feat)

        single_loss = self.ce(outputs, targets) + self.ce(bi_outputs, targets)
        aux_targets = torch.stack([targets * 4 + i for i in range(4)], 1).view(-1)
        joint_loss = self.ce(aux_outputs, aux_targets) + self.ce(biaux_outputs, aux_targets)

        aux_outputs = torch.stack([aux_outputs[i::4, i::4] for i in range(4)], 1).mean(1)
        biaux_outputs = torch.stack([biaux_outputs[i::4, i::4] for i in range(4)], 1).mean(1)

        sla_loss = self.kd(outputs, aux_outputs) + self.kd(bi_outputs, biaux_outputs)
        loss = single_loss + joint_loss + self.aux_lamb * sla_loss

        outputs = torch.cat([outputs, aux_outputs], dim=0)
        bi_outputs = torch.cat([bi_outputs, biaux_outputs], dim=0)
        feats = [torch.cat([f, rf], dim=0) for f, rf in zip(feats, rot_feats)]
        bi_feats = [torch.cat([f, rf], dim=0) for f, rf in zip(bi_feats, rot_bi_feats)]
        return outputs, bi_outputs, feats, bi_feats, loss
    
def att(args, bifpn):
    return Attention(args)


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.p = 2
        self.kd = DistillKL(args)
        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t)
        loss += self.beta * sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])

        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
    

def rand_bbox(size1,size2, lam):
    W = size1
    H = size2
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class gammaCrop(object):
    def __init__(self, scale_min=0.5,scale_max=2):

        self.scale_min = scale_min
        self.scale_max = scale_max


    def __call__(self, sample):
        
        lam = np.random.beta(1, 1)
        lam = max(lam, 1 - lam)
        rand = np.random.uniform(self.scale_min, self.scale_max)
        
        np_img = np.array(sample)
        gamma_data = np.power(np_img, rand)

        bbx1, bby1, bbx2, bby2 = rand_bbox(len(gamma_data),len(gamma_data[0]), lam)
        np_img[bbx1:bbx2, bby1:bby2] = gamma_data[bbx1:bbx2, bby1:bby2]
        
        img_r = Image.fromarray(np_img)
        
        return img_r
    
class gammaAug(object):
    def __init__(self, scale_min=0.5,scale_max=2):

        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):
               
        random_int = random.randint(0, 1)
        if(random_int==0):
            return sample
        else:
            rand = np.random.uniform(self.scale_min, self.scale_max)
            np_img = np.array(sample)
            gamma_data = np.power(np_img, rand)
            img_r = Image.fromarray(np.uint8(gamma_data))
        
            return img_r

class LogAug(object):
    def __init__(self, scale_min=0.5,scale_max=2):

        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):
               
        random_int = random.randint(0, 1)
        if(random_int==0):
            return sample
        else:
            rand = np.random.uniform(self.scale_min, self.scale_max)
            np_img = np.array(sample)
            
            
            gamma_data = rand * np.log1p(np_img+1)
            img_r = Image.fromarray(np.uint8(gamma_data))
        
            return img_r
