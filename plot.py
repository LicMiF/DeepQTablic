from tablicArena import TablicArena
from tablic import Tablic
from players.greedyPlayer import GreedyPlayer


import matplotlib
import matplotlib.pyplot as plt

import threading
import copy

class DeckedTablicArena(TablicArena):
    def simulate_games(self, decks):
        t1_wins, draws, t2_wins = (0, 0, 0)
        total_results = [0, 0]
        for deck in decks:
            battle_results = [0, 0]
            results = self.start_and_play_game(deck,print_plays=False)
            battle_results[0] += results[0]+results[2]
            battle_results[1] += results[1]+results[3]
            self.switch_players()
            results = self.start_and_play_game(deck,print_plays=False)
            battle_results[1] += results[0]+results[2]
            battle_results[0] += results[1]+results[3]
            self.switch_players()
            total_results[0] += battle_results[0]
            total_results[1] += battle_results[1]
            if (battle_results[0] > battle_results[1]): t1_wins += 1
            if (battle_results[0] == battle_results[1]): draws += 1
            if (battle_results[0] < battle_results[1]): t2_wins += 1
        return (t1_wins, draws, t2_wins, total_results[0], total_results[1])


class Tracker:
    def __init__(self, nGames):

        matplotlib.use('Agg') 

        self._data = {}
        self._nGames = nGames
        self._decks=[]
        for _ in range(self._nGames):
            self._decks.append(Tablic.getShuffledDeck())


    def getContext(self):
        return {"data": self._data,
                "nGames": self._nGames,
                "decks": self._decks}
    
    def setContext(self,cntxt):
        self._data=cntxt["data"]
        self._nGames=cntxt["nGames"]
        self._decks=cntxt["decks"]

    def evaluate(self,player,alpha, gamma, episode,path="progress.png"):
        player0 = copy.deepcopy(player)
        thrd=threading.Thread(target=self.evaluateAndPlot,args=(player0,alpha,gamma,episode,path))
        thrd.start()

    def checkAndInit(self,alpha,gamma):

        if alpha not in self._data:
            self._data[alpha]={}
            self._data[alpha][gamma]={}
            self._data[alpha][gamma]["wins"]=[0]
            self._data[alpha][gamma]["pts"]=[0]
            self._data[alpha][gamma]["draws"]=[0]
            self._data[alpha][gamma]["episodes"]=[0]
            return
        
        if gamma not in self._data[alpha]:
            self._data[alpha][gamma]={}
            self._data[alpha][gamma]["wins"]=[0]
            self._data[alpha][gamma]["pts"]=[0]
            self._data[alpha][gamma]["draws"]=[0]
            self._data[alpha][gamma]["episodes"]=[0]
            return



    def evaluateAndPlot(self,player0, alpha, gamma, episode,path="progress.png"):

        alpha=int(alpha*10000)
        gamma=int(gamma*100)

        player1 = GreedyPlayer()
        arena = DeckedTablicArena(player0, player1,player0, player1)
        wins, draws, _, pts, _ = arena.simulate_games(self._decks)

        self.checkAndInit(alpha,gamma)

        self._data[alpha][gamma]["wins"].append(wins)
        self._data[alpha][gamma]["pts"].append(pts/self._nGames)
        self._data[alpha][gamma]["draws"].append(draws)
        self._data[alpha][gamma]["episodes"].append(episode/1000)

        self.plotAndSave(path)

    def plotAndSave(self,path="progress.png"):
        fig,axs=plt.subplots(3,len(self._data))

        for i,(alpha,gammas) in enumerate(self._data.items()):
            for gamma,stats in gammas.items():
                ax=axs[0,i] if len(self._data)!=1 else axs[0]
                ax.plot(stats["episodes"], stats["wins"],label=f'Gamma {gamma}') 
                ax.set_ylabel('Wins')
                ax.set_title(f'Alpha: {alpha/10000}') 
                ax.legend(fontsize='small')
                ax.grid(which='both', linestyle='--', color='gray') 

                ax=axs[1,i] if len(self._data)!=1 else axs[1]
                ax.plot(stats["episodes"], stats["draws"],label=f'Gamma {gamma}') 
                ax.set_ylabel('draws')
                ax.legend(fontsize='small')
                ax.grid(which='both', linestyle='--', color='gray') 

                ax=axs[2,i] if len(self._data)!=1 else axs[2]
                ax.plot(stats["episodes"], stats["pts"],label=f'Gamma {gamma}')
                ax.set_ylabel('Points')
                ax.set_xlabel('Episode * 10^3') 
                ax.legend(fontsize='small')
                ax.grid(which='both', linestyle='--', color='gray') 

        plt.savefig(path, format='png')
