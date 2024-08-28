from tablicArena import TablicArena
from tablic import Tablic
from players.greedyPlayer import GreedyPlayer


import matplotlib
import matplotlib.pyplot as plt

from collections import defaultdict
import threading
import copy

class DeckedTablicArena(TablicArena):
    def simulate_games(self, decks):
        p1_wins, draws, p2_wins = (0, 0, 0)
        total_results = [0, 0]
        for deck in decks:
            battle_results = [0, 0]
            results = self.start_and_play_game(deck,print_plays=False)
            battle_results[0] += results[0]
            battle_results[1] += results[1]
            self.switch_players()
            results = self.start_and_play_game(deck,print_plays=False)
            battle_results[1] += results[0]
            battle_results[0] += results[1]
            self.switch_players()
            total_results[0] += battle_results[0]
            total_results[1] += battle_results[1]
            if (battle_results[0] > battle_results[1]): p1_wins += 1
            if (battle_results[0] == battle_results[1]): draws += 1
            if (battle_results[0] < battle_results[1]): p2_wins += 1
        return (p1_wins, draws, p2_wins, total_results[0], total_results[1])


class Tracker:
    def __init__(self, nGames):

        matplotlib.use('Agg') 

        self._data = defaultdict(lambda: defaultdict(lambda: {"wins": [0], "pts": [0], "draws": [0], "episodes": [0]}))
        self._nGames = nGames
        self._decks=[]
        for _ in range(self._nGames):
            self._decks.append(Tablic.getShuffledDeck())

    def evaluate(self,player,alpha, gamma, episode,path="progress.png"):
        player0 = copy.deepcopy(player)
        thrd=threading.Thread(target=self.evaluateAndPlot,args=(player0,alpha,gamma,episode,path))
        thrd.start()

    def evaluateAndPlot(self,player0, alpha, gamma, episode,path="progress.png"):
        player1 = GreedyPlayer()
        arena = DeckedTablicArena(player0, player1)
        wins, draws, _, pts, _ = arena.simulate_games(self._decks)

        self._data[alpha*10000][gamma*100]["wins"].append(wins)
        self._data[alpha*10000][gamma*100]["pts"].append(pts/self._nGames)
        self._data[alpha*10000][gamma*100]["draws"].append(draws)
        self._data[alpha*10000][gamma*100]["episodes"].append(episode/1000)

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
