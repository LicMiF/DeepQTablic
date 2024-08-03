from itertools import chain,combinations
import numpy as np
import random

class MyTablic:
    def __init__(self,deck=None):
        """
        The cards are represented inside a matrix where row coresponds to diamond,heart,spade and club (0,1,2,3), 
        and columns are the card values in tablic rules, ace can be either 1 or 11
        """
        if deck:
            assert len(deck)==(4*13)
            self._deck=deck
        else:  
            self._deck=self._getShuffledDeck()

        self._table=np.zeros((4,13),dtype=int)
        self._hands=np.zeros((2,4,13),dtype=int)
        self._taken=np.zeros((2,4,13),dtype=int)

        self._isTerminal=False
        self._currentPlayer=0
        self._lastToTake=0

        self._deckPointer=0
        self._handSize=6
        self._tableSize=4
        self._moveCounter=0

        self._rewards=np.zeros(2,dtype=int)

        self._startGame()

    @property
    def isTerminal(self):
        return self._isTerminal

    @property
    def currentPlayer(self):
        return self._currentPlayer

    @property
    def lastToTake(self):
        return self._lastToTake

    @property
    def moveCounter(self):
        return self._moveCounter

    @property
    def rewards(self):
        return self._rewards

    @classmethod
    def _getShuffledDeck(cls):
        deck=[(i,j) for i in range(4) for j in range(13)]
        random.shuffle(deck)
        return deck
        
    @classmethod
    def cardToReward(cls,card):
        (type,value)=card
        value+=1
        
        if type==0 and value==10:
            return 2
            
        if type==3 and value==2:
            return 1
            
        if value==1 or value>9:
            return 1

        return 0



    @classmethod
    def AllAceCombs(cls, hand):
        tmpHand=hand.copy()
        allCombs=[hand]
        while 1 in tmpHand:
            tmpHand[tmpHand.index(1)]=11
            allCombs.append(tmpHand)
        return allCombs

    @classmethod
    def realCardsFromInd(cls,cards):
        if not isinstance(cards, (list)):
            cards=[cards]   
        realCards=[value for (_,value) in cards]
        realCards=[value+1 if value<10 else value+2 for value in realCards]
        return realCards
        
    @classmethod
    def isValidTake(cls,card,take):

        if not take: 
            return True
        
        (_,value)=card
        
        value+=1 if value<10 else 2
        if value==1: value=11
            
        realCards=cls.realCardsFromInd(take)
        allCombs=cls.AllAceCombs(realCards)
        
        for comb in allCombs:
            leftToTake=comb.copy()
            while True:
                currentValue=value
                currentIter=leftToTake.copy()
                for val in currentIter:
                    
                    if currentValue-val>=0:
                        leftToTake.remove(val)
                        currentValue-=val
                        
                    if not leftToTake and currentValue==0:
                        return True
                        
                    if currentValue==0:
                        break  
                        
                if currentValue!=0:
                    break
                    
        return False

    @classmethod
    def allValidTakes(cls,cards,hand):
        allTakes = chain(list(combination) for r in range(0, len(cards)+1) for combination in combinations(cards, r))
        validTakes=[]
        for take in allTakes:
            for card in hand:
                if cls.isValidTake(card,take):
                    if [card,take] not in validTakes:
                        validTakes.append([card,take])
        return validTakes

    def getTableCardsArray(self):
        arr=[(i,j) for i in range(self._table.shape[0]) for j in range(self._table.shape[1]) if self._table[i,j]]
        return arr

    def getHandCardsArray(self,player):
        arr=[(i,j) for i in range(self._hands.shape[1]) for j in range(self._hands.shape[2]) if self._hands[player][i,j]]
        return arr

    def _addToTable(self, cards):
        if not isinstance(cards, (list)):
            cards=[cards]   
        for card in cards:
            (i,j)=card
            self._table[i,j]=1

    def _removeFromTable(self, cards):
        if not isinstance(cards, (list)):
            cards=[cards]   
        for card in cards:
            (i,j)=card
            self._table[i,j]=0

    def _addToHands(self,player,cards):
        if not isinstance(cards, (list)):
            cards=[cards]        
        for card in cards:
            (i,j)=card
            self._hands[player][i,j]=1

    def _removeFromHands(self,player,cards):
        if not isinstance(cards, (list)):
            cards=[cards]
        for card in cards:
            (i,j)=card
            self._hands[player][i,j]=0

    def _addToTaken(self,player,cards):
        if not isinstance(cards, (list)):
            cards=[cards]   
        numOfTakenCards=np.sum(self._taken[player])
        for card in cards:
            self._rewards[player]+=self.cardToReward(card)
            (i,j)=card
            self._taken[player][i,j]=1

        if (numOfTakenCards+len(cards))>=27 and numOfTakenCards<27:
            self._rewards[player]+=3
        

        
    def _dealCards(self):
        playerHand=self._deck[self._deckPointer:self._deckPointer+self._handSize]
        self._addToHands(0,playerHand)
        self._deckPointer=self._deckPointer+self._handSize

        playerHand=self._deck[self._deckPointer:self._deckPointer+self._handSize]
        self._addToHands(1,playerHand)
        self._deckPointer=self._deckPointer+self._handSize
        
    def _startGame(self):
        self._deckPointer+=self._tableSize
        tableCards=self._deck[:self._deckPointer]
        self._addToTable(tableCards)
        self._dealCards()

    def _updateGameState(self):
        if not np.sum(self._hands):
            if self._deckPointer>= len(self._deck):
                self._isTerminal=True
                self._addToTaken(self._lastToTake, self.getTableCardsArray())
                self._removeFromTable(self.getTableCardsArray())
            else:
                self._dealCards()
                
    def playCard(self,card, take):
        assert self.isValidTake(card, take), "The take is not valid"
        assert not self._isTerminal, "The has ended"

        if not take:
            self._removeFromHands(self._currentPlayer,card)
            self._addToTable(card)
        else:
            self._removeFromHands( self._currentPlayer,card)
            self._addToTaken(self._currentPlayer,card)
            self._removeFromTable(take)
            if np.sum(self._table) == 0:
                self._rewards[self._currentPlayer] += 1
            self._addToTaken(self._currentPlayer,take)
            self._lastToTake = self._currentPlayer
        self._currentPlayer = 1 - self._currentPlayer
        self._moveCounter += 1
        self._updateGameState()

    @classmethod
    def fromCardsTo1DMatrix(cls, cards):
        if not cards:
            return np.zeros(4*13,dtype=int)
        if not isinstance(cards, (list)):
            cards=[cards]   
        # print(20* "-")
        # print("cards")
        # print(cards)

        # print([1 if (i,j) in cards else 0 for i in range(4) for j in range(13)])
        # print(20* "-")

        
        return [1 if (i,j) in cards else 0 for i in range(4) for j in range(13)]
        

    def getObservationVector(self,player):
        return np.concatenate((self._table.flatten(),
                               self._hands[player].flatten(),
                               self._taken.flatten(),
                               [self._lastToTake,player]))
        
        
    
        

        
        
        
        