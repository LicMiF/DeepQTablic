from itertools import chain,combinations
from collections import deque
import numpy as np
import random

class Tablic:
    def __init__(self,deck=None):
        """
        The cards are represented inside an array where indexes with  corespond to the card values in tablic rules, ace can be either 1 or 11
        """
        if deck:
            assert len(deck)==(52)
            self._deck=deck
        else:  
            self._deck=self.getShuffledDeck()

        self._table=[]
        self._hands=[[] for _ in range(4)] # 4hands
        self._taken=[[] for _ in range(4)] # 4taken

        self._isTerminal=False
        self._currentPlayer=0 # 4 plyr
        self._lastToTake=0

        self._deckPointer=0
        self._handSize=6
        self._tableSize=4
        self._moveCounter=0

        self._rewards=np.zeros(4,dtype=int) #4rew

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
    
    @property
    def table(self):
        return self._table
    
    @property
    def hands(self):
        return self._hands


    @classmethod
    def getShuffledDeck(cls,seed=None):
        if seed:
            random.seed(seed)
        deck=[cls.indexToCard(j) for _ in range(4) for j in range(13)]
        random.shuffle(deck)
        return deck
        

    @classmethod
    def cardToReward(cls,card):
        if card==1 or card>9:
            return 1
        
        return 0
    

    @classmethod
    def cardToIndex(cls,card):
        if card==11:
            return 0
        
        return card-1 if card<=10 else card-2


    @classmethod
    def indexToCard(cls,index):
        return index+1 if index<10 else index+2


    # Working with card values instead of array indexes, so parameter hand is an array of card values
    @classmethod
    def allAceCombs(cls, hand):
        tmpHand=hand.copy()
        allCombs=[hand]
        while 1 in tmpHand:
            tmpHand[tmpHand.index(1)]=11
            allCombs.append(tmpHand.copy())
        return allCombs


    @classmethod
    def cardsToIndexArray(cls,cards):
        indexArr=np.zeros(13,dtype=int)

        for card in cards:
            indexArr[cls.cardToIndex(card)]+=1

        return indexArr
    
    @classmethod
    def indexArrayToCards(cls, indexArr):
        cards=[]

        for index in range(13):
            for _ in range(indexArr[index]):
                cards.append(cls.indexToCard(index))

        return cards
        
    # Two possibilities for validating a take, one combinatorial and one using BFS, BFS empiricaly slightly faster
    @classmethod
    def isValidTake(cls,card,take):


        def isTotalLenEq(combination,comb):
            totLen=0
            for c in combination:
                totLen+=len(c)
            return totLen==len(comb)
        

        def isSameCardCardinality(combOfCombs,combCardCardinality):
            combOfCombsCardCardinality=np.zeros(14,dtype=int)
            for comb in combOfCombs:
                combCardCardinal=np.zeros(14,dtype=int)
                combCardCardinal=[combCardCardinal[i]+sum([1 for crd in comb if crd==(i+1)]) for i in range(14)]
                combOfCombsCardCardinality+=combCardCardinal
            for i in range(14):
                if combCardCardinality[i]!=combOfCombsCardCardinality[i]:
                    return False
                
            return True
        

        def combinatorial(cmb):
            allSubsets = list(chain(list(combination) for r in range(1, len(cmb)+1) for combination in combinations(cmb, r) if sum(combination)==card))


            combCardCardinality=np.zeros(14,dtype=int)
            combCardCardinality=[combCardCardinality[i]+sum([1 for crd in cmb if crd==(i+1)]) for i in range(14)]


            for r in range(1, len(allSubsets)+1):
                for combination in combinations(allSubsets, r):
                    if isTotalLenEq(combination,cmb) and isSameCardCardinality(combination,combCardCardinality):
                        return True


        def backtrack(cmb):
            dq=deque([cmb])
            while True:
                if len(dq)==0:
                    return False
                
                cards=dq.popleft()

                for r in range(1, len(cards)+1):
                    for combination in combinations(cards, r):
                        if sum(combination)==card:
                            if len(combination)==len(cards):
                                return True
                            
                            cardsCp=cards.copy()
                            for crd in combination:
                                cardsCp.remove(crd)

                            dq.append(cardsCp)


        if not take: 
            return True
        
        if card==1: card=11

        if not isinstance(take, (list)):
            take=list(take)   

        allCombs=cls.allAceCombs(take)
                
        for comb in allCombs:
            comb=np.array(sorted(comb))

            if(np.sum(comb)==card): return True

            if card>=np.max(comb) and (np.sum(comb) % card == 0):
                # Uncomment for backtrack search
                if backtrack(comb.tolist()):
                    return True

                # # Uncomment for combinatorial search
                # if combinatorial(comb):
                #     return True

        return False

    @classmethod
    def allValidTakes(cls,cards,hand):
        allTakes = chain(list(combination) for r in range(0, len(cards)+1) for combination in combinations(cards, r))
        allUniqueTakes = set(tuple(sorted(comb)) for comb in allTakes)
        validTakes=[]
        for take in allUniqueTakes:
            for card in set(hand):
                if cls.isValidTake(card,take):
                    validTakes.append([card,take])
        return validTakes

    @classmethod
    def allValidStateActions(cls,cards,hand,stateRepresentation):
        allTakes = chain(list(combination) for r in range(0, len(cards)+1) for combination in combinations(cards, r))
        allUniqueTakes = set(tuple(sorted(comb)) for comb in allTakes)
        validStateActions=[]
        for take in allUniqueTakes:
            for card in set(hand):
                if cls.isValidTake(card,take):
                    validStateActions.append(
                        np.concatenate(
                        (cls.getActionRepresentation(card,take),
                         stateRepresentation)))
        return validStateActions
    
    @classmethod
    def getActionRepresentation(cls,card, take):
        return np.concatenate((cls.cardsToIndexArray(take),
        cls.cardsToIndexArray([card])))
    
    @classmethod
    def getCardTakeFromActionRepresentation(cls,action):
        return [cls.indexArrayToCards(action[13:])[0],
        tuple(cls.indexArrayToCards(action[:13]))]


    def isTabla(self,take):
        if sorted(take)==sorted(self._table):
            return True
        return False

    def _addToTable(self, cards):
        if not isinstance(cards, (list)):
            cards=[cards]  

        self._table.extend(cards)

    def _removeFromTable(self, cards):
        if not isinstance(cards, (list)):
            cards=[cards]  

        for card in cards:
            self._table.remove(card)


    def _addToHands(self,player,cards):
        if not isinstance(cards, (list)):
            cards=[cards]  

        self._hands[player].extend(cards)

    def _removeFromHands(self,player,cards):
        if not isinstance(cards, (list)):
            cards=[cards]  

        for card in cards:
            self._hands[player].remove(card)

    def _addToTaken(self,player,cards):
        if not isinstance(cards, (list)):
            cards=[cards]   

        numOfTakenCards=len(self._taken[player])

        for card in cards:
            self._rewards[player]+=self.cardToReward(card)
            self._taken[player].append(card)

        if (numOfTakenCards+len(cards))>=27 and numOfTakenCards<27:
            self._rewards[player]+=3
        

    # Here add loop for 4 players
    def _dealCards(self):
        for i in range(4):
            playerHand=self._deck[self._deckPointer:self._deckPointer+self._handSize]
            self._addToHands(i,playerHand)
            self._deckPointer=self._deckPointer+self._handSize
        

    def _startGame(self):
        self._deckPointer+=self._tableSize
        tableCards=self._deck[:self._deckPointer]
        self._addToTable(tableCards)
        self._dealCards()


    def _updateGameState(self):
        if not sum([np.sum(hand) for hand in self._hands]): # just sum all hands (self._hands)?
            if self._deckPointer>= len(self._deck):
                self._isTerminal=True
                self._addToTaken(self._lastToTake, self._table)
                self._removeFromTable(self._table.copy())

            else:
                self._dealCards()


    def playCard(self,card, take):
        assert self.isValidTake(card, take), "The take is not valid"
        assert not self._isTerminal, "The game has ended"

        prePlayChecksum=self._computeChecksum()

        if not take:
            self._removeFromHands(self._currentPlayer,card)
            self._addToTable(card)
        else:
            self._removeFromHands( self._currentPlayer,card)
            self._addToTaken(self._currentPlayer,card)
            self._removeFromTable(take)
            if len(self._table) == 0:
                self._rewards[self._currentPlayer] += 1

            self._addToTaken(self._currentPlayer,take)
            self._lastToTake = self._currentPlayer
        self._currentPlayer = (self._currentPlayer+1) % 4 # CHANGE 
        self._moveCounter += 1

        assert prePlayChecksum==self._computeChecksum(), "Unexpected behaviour"

        self._updateGameState()

    # Used for sanity check
    def _computeChecksum(self):
        totSum=0
        for chckSumIncldd in [self._table, self._hands, self._taken]:
            if not chckSumIncldd:
                continue

            if isinstance(chckSumIncldd[0], (list)):
                totSum+=sum([np.sum(lst) for lst in chckSumIncldd])

            else:
                totSum+=np.sum(chckSumIncldd)
        return totSum
        

    def getEveryPlayerTaken(self,player=None):
         curr=self._currentPlayer
         if player:
             curr=player
         players=[(plyr)%4 for plyr in range(curr,curr+4)]
         return np.concatenate([self.cardsToIndexArray(self._taken[player]) for player in players])

    def getPlayersStateRepresentation(self,player):
        return np.concatenate((self.cardsToIndexArray(self._table),
                               self.cardsToIndexArray(self._hands[player]),
                               self.getEveryPlayerTaken(player),
                               [self._lastToTake,player]))
    

    # Making independent of currentPlayer, fixed order of taken ?
    def getGameStateRepresentation(self):
        return np.concatenate((self.cardsToIndexArray(self._table),
                               self.cardsToIndexArray(self._hands[self._currentPlayer]),
                               self.getEveryPlayerTaken(),
                               [self._lastToTake,self._currentPlayer]))
    

    # loop over 4 players
    def displayGameInfo(self):
        print(f"Table: {self._table}")
        print(50*"-")
        for i in range(4):
            print(f"Player {i+1} hand: {self._hands[self._currentPlayer]}")

            print(f"Player {i+1} taken: {self._taken[self._currentPlayer]}")

            print(f"Player {i+1} reward: {self._rewards[self._currentPlayer]}")

            print(50*"-")

        print(f"Current player: {self._currentPlayer}, last to take: {self._lastToTake}, deck pointer: {self._deckPointer}")
        print()
        print()