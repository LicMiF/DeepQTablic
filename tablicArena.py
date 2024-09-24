from tablic import Tablic
from players.greedyPlayer import GreedyPlayer
from players.qPlayer import QPlayer
import random

class TablicArena:
    def __init__(self, player1, player2, player3, player4):
        self.player1 = player1
        self.player2 = player2
        self.player3 = player3
        self.player4 = player4
    
    def simulate_games(self, num_of_games, printPlays=False ,seed=None):
        random.seed(seed)
        t1_wins, draws, t2_wins = (0, 0, 0)
        total_results = [0, 0]
        for game in range(1, num_of_games+1):
            deck = Tablic.getShuffledDeck()
            battle_results = [0, 0]
            results = self.start_and_play_game(deck,print_plays=printPlays)
            battle_results[0] += results[0]+results[2]
            battle_results[1] += results[1]+results[3]
            self.switch_players()
            results = self.start_and_play_game(deck,print_plays=printPlays)
            battle_results[1] += results[0]+results[2]
            battle_results[0] += results[1]+results[3]
            self.switch_players()
            total_results[0] += battle_results[0]
            total_results[1] += battle_results[1]
            if (battle_results[0] > battle_results[1]): t1_wins += 1
            if (battle_results[0] == battle_results[1]): draws += 1
            if (battle_results[0] < battle_results[1]): t2_wins += 1
        return (t1_wins, draws, t2_wins, total_results[0], total_results[1])

    def start_and_play_game(self, deck=None, print_plays=False):
        game = Tablic(deck)
        while not game.isTerminal:
            for player in [self.player1, self.player2, self.player3, self.player4]:
                card, take = player.play(game)
                if (print_plays): print(f"Playing {card} and taking {take}.")
                game.playCard(card, list(take))
        return game.rewards
    
    def switch_players(self):
        self.player1, self.player2 = self.player2, self.player1
        self.player3, self.player4 = self.player4, self.player3



if __name__ == '__main__':
    BATTLES = 100
    # player0 = QPlayer()
    # player0.loadModelParams("models/minimalModelParams/g10e49000")
    player0=GreedyPlayer()
    player1 = GreedyPlayer()
    # player1 = GreedyPlayer(True)
    arena = TablicArena(player0, player1,player0, player1)
    wins, draws, loses, pts, opp_pts = arena.simulate_games(BATTLES,False, 0)
    print(f"W:{wins}, D:{draws}, L:{loses}")
    print(pts / BATTLES, opp_pts / BATTLES)
    print()