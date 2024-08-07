from tablic import Tablic
from players.greedyPlayer import GreedyPlayer
import random

class TablicArena:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
    
    def simulate_games(self, num_of_games, printPlays=False ,seed=None):
        random.seed(seed)
        p1_wins, draws, p2_wins = (0, 0, 0)
        total_results = [0, 0]
        for game in range(1, num_of_games+1):
            deck = Tablic.getShuffledDeck()
            battle_results = [0, 0]
            results = self.start_and_play_game(deck,print_plays=printPlays)
            battle_results[0] += results[0]
            battle_results[1] += results[1]
            self.switch_players()
            results = self.start_and_play_game(deck,print_plays=printPlays)
            battle_results[1] += results[0]
            battle_results[0] += results[1]
            self.switch_players()
            total_results[0] += battle_results[0]
            total_results[1] += battle_results[1]
            if (battle_results[0] > battle_results[1]): p1_wins += 1
            if (battle_results[0] == battle_results[1]): draws += 1
            if (battle_results[0] < battle_results[1]): p2_wins += 1
        return (p1_wins, draws, p2_wins, total_results[0], total_results[1])

    def start_and_play_game(self, deck=None, print_plays=False):
        game = Tablic(deck)
        while not game.isTerminal:
            card, take = self.player1.play(game)
            if (print_plays): print(f"Playing {card} and taking {take}.")
            game.playCard(card, list(take))
            card, take = self.player2.play(game)
            if (print_plays): print(f"Playing {card} and taking {take}.")
            game.playCard(card, list(take))
        return game.rewards
    
    def switch_players(self):
        self.player1, self.player2 = self.player2, self.player1



if __name__ == '__main__':
    BATTLES = 20
    player0 = GreedyPlayer()
    player1 = GreedyPlayer()
    arena = TablicArena(player0, player1)
    wins, draws, loses, pts, opp_pts = arena.simulate_games(BATTLES,True, 0)
    print(f"W:{wins}, D:{draws}, L:{loses}")
    print(pts / BATTLES, opp_pts / BATTLES)
    print()