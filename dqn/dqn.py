import math
import torch
from torch import nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)

class DQN(nn.Module):
    # # Uncomment for half floating point precision
    # def __init__(self):
    #     super().__init__()
    #     inputSize = 80
    #     outputSize = 1
    #     self.reluStack = nn.Sequential(
    #         nn.Linear(inputSize, inputSize*2).half(),
    #         nn.ReLU(),
    #         nn.Linear(inputSize*2, inputSize*2).half(),
    #         nn.ReLU(),
    #         nn.Linear(inputSize*2, outputSize).half())

    def __init__(self):
        super().__init__()
        inputSize = 80
        outputSize = 1
        self.reluStack = nn.Sequential(
            nn.Linear(inputSize, inputSize*2),
            nn.ReLU(),
            nn.Linear(inputSize*2, inputSize*2),
            nn.ReLU(),
            nn.Linear(inputSize*2, outputSize))

    def forward(self, x):
        return self.reluStack(x)
    
class DQNnoisy(nn.Module):
    def __init__(self):
        super().__init__()
        inputSize = 80
        outputSize = 1
        self.reluStack = nn.Sequential(
            nn.Linear(inputSize, inputSize*2),
            nn.ReLU(),
            nn.Linear(inputSize*2, inputSize*2),
            nn.ReLU(),
            NoisyLinear(inputSize*2, outputSize))

    def forward(self, x):
        return self.reluStack(x)
    
    
class ResidualNoisyDeepDQN(nn.Module):
    def __init__(self):
        super().__init__()
        inputSize = 80
        outputSize = 1
        self.fc1 = nn.Linear(inputSize, inputSize*4)
        self.fc2 = nn.Linear(inputSize*4, inputSize*4)
        self.fc3 = nn.Linear(inputSize*4, inputSize*2)
        self.fc4 = NoisyLinear(inputSize*2, outputSize)
        self.relu = nn.ReLU()

    def forward(self, x):        
        x = self.relu(self.fc1(x))

        residual = x

        x = self.relu(self.fc2(x))
        
        x = x + residual
        
        x = self.relu(self.fc3(x))

        x = self.fc4(x)
        
        return x
    
    
def getArchitecture(name):
    if name=="resNoisy":
        return ResidualNoisyDeepDQN()
    if name=="Noisy":
        return DQNnoisy()
    if name=="Initial":
        return DQN()
