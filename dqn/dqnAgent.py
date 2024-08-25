from dqn.dqn import DQN
from dqn.replayBuffer import ReplayBuffer
import numpy as np
import torch

class DQNAgent():

    def __init__(self,gamma=0,replBufferSize=1024*16,miniBatchSize=1024,alpha=0.001, device="cpu"):
        
        if device=="cuda" and not torch.cuda.is_available():
            raise SystemExit("Selected device is not available on current machine")
        
        self.device=device

        self.setTorchSeed(0)

        self.target = DQN().to(self.device)
        self.current = DQN().to(self.device)


        # self.gamma = torch.tensor(gamma).reshape((1,1))
        self.gamma=gamma
        self.miniBatchSize = miniBatchSize
        self.buffer=ReplayBuffer(size=replBufferSize)

        self.lossFn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.current.parameters(),
                                            lr=alpha)

    @classmethod
    def getAvailDevice(cls):
        device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        return device
    
    @classmethod
    def setTorchSeed(cls,seed):
        torch.manual_seed(seed)  
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed) 
            torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.current(x.to(self.device))

    def saveModelsParams(self,path):
        torch.save(self.current.state_dict(), path+"Curr.pth")
        torch.save(self.target.state_dict(), path+"Target.pth")

    def loadModelsParams(self,path):
        self.current.load_state_dict(torch.load(path+"Curr.pth"))
        self.target.load_state_dict(torch.load(path+"Target.pth"))

    def updateTarget(self):
        self.target.load_state_dict(self.current.state_dict())

    def remember(self, SA, reward, nextStateSAs):
        self.buffer.add((torch.tensor(SA,dtype=torch.float32), reward,torch.tensor(np.array(nextStateSAs),dtype=torch.float32)))

    def backward(self):
        miniBatch = self.buffer.sample(self.miniBatchSize)
        currentSAmb = torch.zeros([len(miniBatch), 80])

        for ind, (SA, _, _) in enumerate(miniBatch):
            currentSAmb[ind] = SA

        qVals = self.current.forward(currentSAmb.to(self.device))
        with torch.no_grad():
            targetQVals = qVals.clone().detach()
            for ind, (_, reward, nextStateSAs) in enumerate(miniBatch):
                if nextStateSAs.numel() == 0:
                    targetQVals[ind] = reward
                else:
                    targetQVals[ind] = reward + self.gamma * torch.max(self.target(nextStateSAs.to(self.device)))
        
        loss = self.lossFn(qVals, targetQVals)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



