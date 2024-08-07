from dqnAgent import DQNAgent
from tablic import Tablic
import random
import torch
from torch import Tensor
from replayBuffer import ReplayBuffer


def initTests():
    random.seed(0)
    agent=DQNAgent()
    print(agent.device)
    print(agent.gamma)
    print(agent.miniBatchSize)
    print(agent.buffer)
    print(agent.target)
    print(agent.current)
    print(agent.lossFn)
    print(agent.optimizer)

    print()
    print()
    print()

    print("Model parameters (weights and biases):")
    for name, param in agent.current.named_parameters():
        print(f"{name}: {param}")
    
    print()
    print()
    print()

    print("Model parameters (weights and biases):")
    for name, param in agent.target.named_parameters():
        print(f"{name}: {param}")


def testForward():
    random.seed(0)
    deck=Tablic.getShuffledDeck()
    game=Tablic(deck=deck)
    allValidStateActions=Tablic.allValidStateActions(game.table,game._hands[game.currentPlayer],game.getGameStateRepresentation())
    agent=DQNAgent()
    with torch.no_grad():
        print(torch.tensor(allValidStateActions,dtype=torch.float32))
        print(agent.forward(torch.tensor(allValidStateActions,dtype=torch.float32)))

def testTargetUpdate():
    agent=DQNAgent()
    
    isEqu=True
    for (param1,param2) in zip(agent.target.parameters(),agent.current.parameters()):
        
        isEqu=isEqu and torch.equal(param1, param2)

    print(isEqu)

    agent.updateTarget()

    isEqu=True
    for (param1,param2) in zip(agent.target.parameters(),agent.current.parameters()):
        
        isEqu=isEqu and torch.equal(param1, param2)
    
    print(isEqu)

def testBuffer():
    random.seed(0)
    agent=DQNAgent()
    deck=Tablic.getShuffledDeck()
    game=Tablic(deck=deck)
    while not game.isTerminal:
        allValidTakes=Tablic.allValidTakes(game.table,game._hands[game.currentPlayer])
        allValidStateActions=Tablic.allValidStateActions(game.table,game._hands[game.currentPlayer],game.getGameStateRepresentation())
        
        randPlay=random.choice(allValidTakes)
        
        playInd=allValidTakes.index(randPlay)

        game.playCard(randPlay[0],list(randPlay[1]))

        allValidTakes1=Tablic.allValidTakes(game.table,game._hands[game.currentPlayer])

        randPlay=random.choice(allValidTakes1)

        game.playCard(randPlay[0],list(randPlay[1]))

        allValidStateActions1=Tablic.allValidStateActions(game.table,game._hands[game.currentPlayer],game.getGameStateRepresentation())

        agent.remember(allValidStateActions[playInd],10,allValidStateActions1)

    # print(agent.buffer._buffer)
    print(agent.buffer.sample(agent.miniBatchSize))

def testSamplingFromBuff():
    random.seed(0)
    agent=DQNAgent()
    buff=ReplayBuffer(16*1024)
    for i in range(10000):
        agent.remember(i,i+1,i+2)
    print(agent.buffer.sample(10))
    print(agent.buffer.sample(10))

def testBackward():
    random.seed(0)
    agent=DQNAgent()
    for _ in range(20):
        deck=Tablic.getShuffledDeck()
        game=Tablic(deck=deck)
        while not game.isTerminal:
            allValidTakes=Tablic.allValidTakes(game.table,game._hands[game.currentPlayer])
            allValidStateActions=Tablic.allValidStateActions(game.table,game._hands[game.currentPlayer],game.getGameStateRepresentation())
            
            randPlay=random.choice(allValidTakes)
            
            playInd=allValidTakes.index(randPlay)

            game.playCard(randPlay[0],list(randPlay[1]))

            allValidTakes1=Tablic.allValidTakes(game.table,game._hands[game.currentPlayer])

            randPlay=random.choice(allValidTakes1)

            game.playCard(randPlay[0],list(randPlay[1]))

            allValidStateActions1=Tablic.allValidStateActions(game.table,game._hands[game.currentPlayer],game.getGameStateRepresentation())

            agent.remember(allValidStateActions[playInd],10,allValidStateActions1)
        print(Tablic.allValidStateActions(game.table,game._hands[game.currentPlayer],game.getGameStateRepresentation()))


    preParam=[param.clone() for param in agent.current.parameters()]

    agent.backward()
    
    isEqu=True
    for (param1,param2) in zip(preParam,agent.current.parameters()):
        
        isEqu=isEqu and torch.equal(param1, param2)
    
    print(isEqu)

if __name__=="__main__":
    # initTests() 
    # testForward()
    # testTargetUpdate()
    testBackward()