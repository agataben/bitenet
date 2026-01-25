import torch
import random
import numpy as np


# Seed setting
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self,value,num):
        self.sum += value*num
        self.num += num

    def value(self):
        try:
            return self.sum/self.num
        except:
            return None

def calculate_mean_and_std(x):
    canals_n = x[0][0].shape[0]

    mean = torch.zeros(canals_n)
    for sample in x:
        mean += sample[0].view(canals_n, -1).mean(dim=1)
    mean /= len(x)

    std = torch.zeros(canals_n)
    for sample in x:
        std += ((sample[0].view(canals_n, -1) - mean[:, None])**2).mean(dim=1)
    std = torch.sqrt(std / len(x))

    return mean, std

