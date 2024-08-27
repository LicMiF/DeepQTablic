from tablicArena import TablicArena
from players.qPlayer import QPlayer
from players.greedyPlayer import GreedyPlayer

import matplotlib
import matplotlib.pyplot as plt

from collections import defaultdict
import threading
import copy

class Tracker:
    def __init__(self, nGames):

        matplotlib.use('Agg') 

        self._data = defaultdict(lambda: defaultdict(lambda: {"wins": [0], "pts": [0], "draws": [0], "episodes": [0]}))
        self._nGames = nGames

    def evaluate(self,player,alpha, gamma, episode):
        player0 = copy.deepcopy(player)
        thrd=threading.Thread(target=self.evaluateAndPlot,args=(player0,alpha,gamma,episode))
        thrd.start()

    def evaluateAndPlot(self,player0, alpha, gamma, episode):
        player1 = GreedyPlayer()
        arena = TablicArena(player0, player1)
        wins, draws, _, pts, _ = arena.simulate_games(self._nGames, False, 0)

        self._data[alpha*10000][gamma*100]["wins"].append(wins)
        self._data[alpha*10000][gamma*100]["pts"].append(pts/self._nGames)
        self._data[alpha*10000][gamma*100]["draws"].append(draws)
        self._data[alpha*10000][gamma*100]["episodes"].append(episode/1000)

        self.plotAndSave()

    def plotAndSave(self):
        fig,axs=plt.subplots(2,len(self._data))

        for i,(alpha,gammas) in enumerate(self._data.items()):
            for gamma,stats in gammas.items():
                ax=axs[0,i] if len(self._data)!=1 else axs[0]
                ax.plot(stats["episodes"], stats["wins"],label=f'Gamma {gamma}') 
                ax.set_ylabel('Wins')
                ax.set_title(f'Alpha: {alpha/10000}') 
                ax.legend(fontsize='small')
                ax.grid(which='both', linestyle='--', color='gray') 

                ax=axs[1,i] if len(self._data)!=1 else axs[1]
                ax.plot(stats["episodes"], stats["pts"],label=f'Gamma {gamma}')
                ax.set_ylabel('Points')
                ax.set_xlabel('Episode * 10^3') 
                ax.legend(fontsize='small')
                ax.grid(which='both', linestyle='--', color='gray') 

        plt.savefig('progress.png', format='png')



