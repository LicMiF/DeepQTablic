import torch
from torch import nn

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