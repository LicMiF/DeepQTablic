from player import Player
from tablic import Tablic
import numpy as np

class GreedyPlayer(Player):

    """Player that maximizes temporal reward and takes tabla into account.
    Empirically shown that keeping tricks in hand beats randomized card
    play when there is no take"""

    def __init__(self,keepTricks=False):
        super().__init__()
        self._keepTricks=keepTricks

    def calcRewards(self,play,game):
        card,take=play

        if not take:
            if self._keepTricks:
                # Try to leave tricks in hand
                return (-Tablic.cardToReward(card),1)
            else:
                # Randomize which card gets out on table if no take
                return (0,1)
        
        totReward=Tablic.cardToReward(card)

        for crd in take:
            totReward+=Tablic.cardToReward(crd)

        totReward+=int(game.isTabla(take))

        return (totReward,len(take)+1)

    def findGreediestPlay(self,allValidPlays,game):
        greediestPlay=max(allValidPlays,key=lambda play: self.calcRewards(play,game))
        
        return greediestPlay

    def playerPolicyPlay(self, game):
        allValidPlays=game.allValidTakes(game.table,game.hands[game.currentPlayer])
        return self.findGreediestPlay(allValidPlays,game)