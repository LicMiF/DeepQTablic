from dqn.dqn import DQN
from dqn.replayBuffer import ReplayBuffer
from dqn.prioritizedMem import Memory
import numpy as np
import torch
from collections import deque


class DQNAgent():

    def __init__(self,gamma,nStepSize,replBufferSize=1024*16,miniBatchSize=1024,alpha=0.001, device="cpu",prioritized=False,multiStep=False):
        
        if device=="cuda" and not torch.cuda.is_available():
            raise SystemExit("Selected device is not available on current machine")
        
        self.device=device
        self.prioritized=prioritized
        self.multiStep=multiStep
        self.nStepSize=nStepSize
        self.nStepBuffer=deque(maxlen=self.nStepSize)

        

        self.setTorchSeed(0)

        self.target = DQN().to(self.device)
        self.current = DQN().to(self.device)


        # self.gamma = torch.tensor(gamma).reshape((1,1))
        self.gamma=gamma
        self.miniBatchSize = miniBatchSize

        if self.prioritized:
            self.buffer=Memory(capacity=replBufferSize)
        else:
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

    def saveCheckpoint(self,trainDict,path):
        trainDict["curr"]=self.current.state_dict()
        trainDict["target"]=self.target.state_dict()
        trainDict["optimizer"]=self.optimizer.state_dict()
        trainDict["buffer"]=self.buffer
        trainDict["gamma"]=self.gamma
        trainDict["device"]=self.device
        trainDict["miniBatchSize"]=self.miniBatchSize
        trainDict["torchRngState"]= torch.get_rng_state()
        if self.device == "cuda":
            trainDict["cudaRngState"]=torch.cuda.get_rng_state_all() 
        torch.save(trainDict,path)

    def loadModelsParams(self,path):
        self.current.load_state_dict(torch.load(path+"Curr.pth"))
        self.target.load_state_dict(torch.load(path+"Target.pth"))

    def loadCheckpoint(self, path):
        trainDict=torch.load(path)
        self.current.load_state_dict(trainDict["curr"])
        self.target.load_state_dict(trainDict["target"])
        self.optimizer.load_state_dict(trainDict["optimizer"])
        self.buffer=trainDict["buffer"]
        self.gamma=trainDict["gamma"]
        self.device=trainDict["device"]
        self.miniBatchSize = trainDict["miniBatchSize"]
        torch.set_rng_state(trainDict['torchRngState'])
        if self.device == "cuda":
            torch.cuda.set_rng_state_all(trainDict['cudaRngState'])
        return trainDict

    def updateTarget(self):
        self.target.load_state_dict(self.current.state_dict())

    def calculateError(self, SA, reward, nextStateSAs):
        with torch.no_grad():
            qVal = self.current.forward(SA.to(self.device))
            if nextStateSAs.numel() == 0:
                targetQVal = reward
            else:
                targetQVal = reward + (self.gamma ** self.nStepSize) * torch.max(self.target(nextStateSAs.to(self.device)))
            
            return torch.abs(qVal-targetQVal).data.numpy()
        
    def rememberNstep(self, SA, reward, nextStateSAs):
        self.nStepBuffer.append((SA,reward,nextStateSAs))
        if len(self.nStepBuffer)<self.nStepSize:
            return
  
        _,lr,lnSAs=self.nStepBuffer[-1]
        for (sa,r,nSAs) in reversed(list(self.nStepBuffer)[:-1]):
            lr=r+self.gamma*lr if nSAs.numel()!=0 else r
            lnSAs=lnSAs if nSAs.numel()!=0 else nSAs

        lsa, _, _ = self.nStepBuffer[0]

        self.buffer.add((lsa,lr,lnSAs))


    
    def remember(self, SA, reward, nextStateSAs):
        sa,rew,nSAs=(torch.tensor(SA,dtype=torch.float32), reward,torch.tensor(np.array(nextStateSAs),dtype=torch.float32))

        if self.multiStep:
            self.rememberNstep(sa,rew,nSAs)
            return
        
        if self.prioritized:
            error=self.calculateError(sa,rew,nSAs)
            self.buffer.add(error, (sa,rew,nSAs))
        else:
            self.buffer.add((sa,rew,nSAs))

    def backward(self):
        if self.prioritized:
            miniBatch, idxs, isWeights = self.buffer.sample(self.miniBatchSize)
        else:
            miniBatch = self.buffer.sample(self.miniBatchSize)

        # print(miniBatch)

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
                    targetQVals[ind] = reward + (self.gamma**self.nStepSize) * torch.max(self.target(nextStateSAs.to(self.device)))
        
            if self.prioritized:
                errors=torch.abs(qVals-targetQVals).data.numpy()

                # update priority
                for i in range(len(miniBatch)):
                    idx = idxs[i]
                    self.buffer.update(idx, errors[i])

        loss = self.lossFn(qVals, targetQVals)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



