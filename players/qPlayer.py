from players.player import Player
from dqn.dqnAgent import DQNAgent
import torch
import numpy as np
import random

class QPlayer(Player):
    def __init__(self,agent=DQNAgent()):
        super().__init__()
        self.agent=agent

    def loadModelParams(self,path):
        self.agent.loadModelsParams(path)

    def saveModelParams(self,path):
        self.agent.saveModelsParams(path)

    def playerRandomPlay(self, game):
        allValidPlays=game.allValidTakes(game.table,game.hands[game.currentPlayer])
        return random.choice(allValidPlays)

    def playerPolicyPlay(self, game):
        allValidStateActions=game.allValidStateActions(game.table,game.hands[game.currentPlayer],game.getGameStateRepresentation())
        with torch.no_grad():
            allQVals=self.agent.forward(torch.tensor(np.array(allValidStateActions),dtype=torch.float32))
        maxQsaInd=torch.argmax(allQVals)
        return game.getCardTakeFromActionRepresentation(allValidStateActions[maxQsaInd][:26])

