

def checkDeckShuffling(seed=None):

    random.seed(seed)
    deck=Tablic.getShuffledDeck()
    occurances=[0 for _ in range (13)]
    for i in range(13):
        for card in deck:
            if i==card:
                occurances[i]+=1
    print(deck)
    print(len(deck))
    assert sum(occurances)==sum([4 for _ in range(13)])

def checkToReward():
    for i in range(13):
        print(Tablic.cardToReward(i))


def checkIndexToCard():
    cards=[]
    for i in range(13):
        cards.append(Tablic.indexToCard(i))
    print(cards)
    return cards

def checkCardToIndex():
    indexes=[]
    cards=checkIndexToCard()
    for card in cards:
        indexes.append(Tablic.cardToIndex(card))
    print(indexes)


def checkCardToIndexArray():
    cards=checkIndexToCard()
    indexes=Tablic.cardsToIndexArray(cards)
    print(indexes)
    return indexes+1

def checkIndexArrayToCards():
    indexes=checkCardToIndexArray()
    cards=Tablic.indexArrayToCards(indexes)
    print(cards) 
    return cards


def checkTablicGameStart():
    deck=Tablic.getShuffledDeck()
    print(deck)
    game=Tablic()
    game.displayGameInfo()


def checkEksplicitDeck():
    deck=Tablic.getShuffledDeck()
    print(deck)
    game=Tablic(deck)
    game.displayGameInfo()

        
def simulateGame():
    deck=Tablic.getShuffledDeck(5)
    game=Tablic(deck=deck)
    game.displayGameInfo()
    while not game.isTerminal:
        tm=time.time()
        allValidTakes=Tablic.allValidTakes(game._table,game._hands[game.currentPlayer])
        # allValidTakes=list(allValidTakes)
        print(f"Elapsed{time.time()-tm}")
        print(allValidTakes)
        randPlay=random.choice(allValidTakes)
        game.playCard(randPlay[0],randPlay[1])
        # print()
        # print(f"All valid takes:{allValidTakes}")
        # print(f"Played move{randPlay}")
        # print()
        # game.displayGameInfo()

def simulateGameNkili():
    deck=NkiliTablic.get_shuffled_deck(5)
    game=NkiliTablic(deck=deck)
    while not game.is_terminal:
        tm=time.time()
        allValidTakes=NkiliTablic.get_valid_takes(game.table,game._hands[game.current_player])
        allValidTakes=list(allValidTakes)
        print(f"Elapsed{time.time()-tm}")
        print(allValidTakes)
        randPlay=random.choice(allValidTakes)
        game.play_card(randPlay[0],randPlay[1])
        # print()
        # print(f"All valid takes:{allValidTakes}")
        # print(f"Played move{randPlay}")
        # print()
        # game.displayGameInfo()

def compareLists(ls,ls1):
    lsC=ls.copy()
    ls1C=ls1.copy()

    for el in ls:
        for el1 in ls1:
            if el[0]==el1[0] and (sorted(list(el[1]))==sorted(list(el1[1]))):
                lsC=[lsC[i] for i in range(len(lsC)) if lsC[i]!=el]
                ls1C=[ls1C[i] for i in range(len(ls1C)) if ls1C[i]!=el1]
    # print(lsC)
    # print(ls1C)
    return not lsC and (not ls1C)

# for 
def compareAllValidTakes(rng):
    for i in range(rng):
        deck=Tablic.getShuffledDeck(i)
        game=Tablic(deck=deck)
        while not game.isTerminal:
            tm=time.time()
            allValidTakes=Tablic.allValidTakes(game._table,game._hands[game.currentPlayer])
            allValidTakes=list(allValidTakes)
            allValidTakes1=NkiliTablic.get_valid_takes(game._table,game._hands[game.currentPlayer])
            allValidTakes1=list(allValidTakes1)
            if not compareLists(allValidTakes,allValidTakes1):
                print("Not ok")
                maks=max([len(take[1]) for take in allValidTakes])
                print(maks)
                pprinted=[take for i in range(maks+1) for take in allValidTakes if i==len(take[1])]
                print(pprinted)
                print(allValidTakes1)
                return
            # allValidTakes=list(allValidTakes)
            # print(f"Elapsed{time.time()-tm}")
            # print(allValidTakes)
            randPlay=random.choice(allValidTakes)
            game.playCard(randPlay[0],list(randPlay[1]))
            # print()
            # print(f"All valid takes:{allValidTakes}")
            # print(f"Played move{randPlay}")
            # print()
            # game.displayGameInfo()
        print(f"All ok, seed: {i}")

def measureValidTakesPerformanceForEach():
    measureValidTakesPerformance(lambda cards, hand: Tablic.allValidTakes(cards, hand))
    measureValidTakesPerformance(lambda cards, hand: list(NkiliTablic.get_valid_takes(cards, hand)))

def measureValidTakesPerformance(validTakesFunc):
    totTime=0
    for i in range(100):
        deck=Tablic.getShuffledDeck(i)
        game=Tablic(deck=deck)
        tm=time.time()
        while not game.isTerminal:
            allValidTakes=validTakesFunc(game._table,game._hands[game.currentPlayer])
            randPlay=random.choice(allValidTakes)
            game.playCard(randPlay[0],list(randPlay[1]))
        print(f"Elapsed{time.time()-tm} iteration {i}")
        totTime+=time.time()-tm
    print(f"Total average time is: {totTime/100}")

def testObservationVectors():
    deck=Tablic.getShuffledDeck(5)
    game=Tablic(deck=deck)
    game.displayGameInfo()
    while not game.isTerminal:
        allValidTakes=Tablic.allValidTakes(game._table,game._hands[game.currentPlayer])
        randPlay=random.choice(allValidTakes)
        game.playCard(randPlay[0],list(randPlay[1]))
        observation=game.getGameObservationVector()[:-2]
        totSum=0
        for i in range (4):
            totSum+=sum(game.indexArrayToCards(game.getGameObservationVector()[i*13:(i+1)*13]))
        if (game._computeChecksum()-sum(game._hands[1-game.currentPlayer]))==totSum:
            print("All Ok")
        else:
            print("There is a missmatch")
        # game.displayGameInfo()

def checkIsTabla():
    deck=Tablic.getShuffledDeck(5)
    game=Tablic(deck=deck)
    game.displayGameInfo()
    print(game.isTabla(game._table+[2]))

def initWorkDir():
    import sys
    import os

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)


if __name__=="__main__":
    initWorkDir()

    from benchmarking.nkiliTablic import NkiliTablic
    import random
    import time
    import numpy as np
    from tablic import Tablic

    # measureValidTakesPerformanceForEach()
    compareAllValidTakes(1000)
    # print(Tablic.isValidTake(13, (6, 4, 5, 5, 3, 3)))
    # testObservationVectors()
    # checkIsTabla()


