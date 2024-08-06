import numpy as np
import random
from itertools import chain, combinations


class NkiliTablic:
    @classmethod
    def card_to_tricks(cls, card):
        return 1 if card == 1 or card >= 10 else 0

    @classmethod
    def card_to_index(cls, card):
        if card  == 11:
            return cls.card_to_index(1)
        if card > 11:
            return card - 2
        return card - 1

    @classmethod
    def index_to_card(cls, index):
        card = index + 1
        if card > 10:
            card = card + 1
        return card

    @classmethod
    def _get_all_ace_combinations(cls, take):
        take = list(take)
        yield take.copy()
        while 1 in take:
            take[take.index(1)] = 11
            yield take.copy()

    @classmethod
    def _find_subset(cls, size, elements, subset):
        if size == 0:
            return subset
        if size < 0 or len(elements) == 0:
            return None
        found_subset = cls._find_subset(size - elements[0], elements[1:], subset + [elements[0], ])
        if found_subset:
            return found_subset
        return cls._find_subset(size, elements[1:], subset)

    @classmethod
    def is_valid_take(cls, played_card, take):
        if len(take) == 0: return True
        if played_card == 1: played_card = 11
        for take in cls._get_all_ace_combinations(take):
            left_to_take = sorted(take, reverse=True)
            while left_to_take:
                subset = cls._find_subset(played_card, left_to_take, [])
                if subset == None:
                    break
                for element in subset:
                    left_to_take.remove(element)
            if not left_to_take:
                return True
        return False

    @classmethod
    def get_valid_takes(cls, table, hand):
        all_takes = chain.from_iterable(set(combinations(table, r)) for r in range(0, len(table)+1))
        for take in all_takes:
            for played_card in hand:
                if NkiliTablic.is_valid_take(played_card, take):
                    yield (played_card, take)


    #Added only seed setting option
    @classmethod
    def get_shuffled_deck(cls,seed=None):
        if seed:
            random.seed(seed)
        deck = [1,2,3,4,5,6,7,8,9,10,12,13,14] * 4
        random.shuffle(deck)
        return deck

    def __init__(self, deck=None, hand_size=6, initial_table_size=4):
        if (deck == None):
            self._deck = self.get_shuffled_deck()
        else:
            self._deck = deck.copy()
        self._hand_size = hand_size
        self._deck_pointer = 0
        self._current_player = 0
        self._last_to_take = 0
        self._move_counter = 0
        self._is_terminal = False
        self._table = []
        self._observation_table = np.zeros(13)
        self._taken = [[], []]
        self._observation_taken = np.zeros([2, 13])
        self._hands = [[], []]
        self._observation_hands = np.zeros([2, 13])
        self._rewards = np.zeros(2)
        self._start_game(initial_table_size)

    @property
    def current_player(self):
        return self._current_player

    @property
    def last_to_take(self):
        return self._last_to_take
    
    @property
    def is_terminal(self):
        return self._is_terminal

    @property
    def move_counter(self):
        return self._move_counter

    @property
    def table(self):
        return self._table.copy()
    
    @property
    def rewards(self):
        return self._rewards.copy()

    def get_taken(self, player):
        return self._taken[player].copy()
    
    def get_hand(self, player):
        return self._hands[player].copy()

    def get_observation_vector(self, player):
        return np.concatenate((self._observation_hands[player], self._observation_table,
                    self._observation_taken[player], self._observation_taken[1-player],
                    [self._last_to_take, player]))

    @classmethod
    def get_take_vector(cls, played_card, take):
        take_vector = np.zeros(26)
        take_vector[NkiliTablic.card_to_index(played_card) + 13] = 1
        for card in take:
            take_vector[NkiliTablic.card_to_index(card)] += 1
        return take_vector

    def _add_cards(self, cards, real, observation):
        if not hasattr(cards, '__iter__'): cards = [cards, ]
        for card in cards:
            if (card == 11): card = 1
            real += [card, ]
            observation[self.card_to_index(card)] += 1

    def _remove_cards(self, cards, real, observation):
        if not hasattr(cards, '__iter__'): cards = [cards, ]
        for card in cards:
            if (card == 11): card = 1
            real.remove(card)
            observation[self.card_to_index(card)] -= 1

    def _add_to_table(self, cards):
        self._add_cards(cards, self._table, self._observation_table)

    def _remove_from_table(self, cards):
        self._remove_cards(cards, self._table, self._observation_table)

    def _add_to_hand(self, cards, player):
        self._add_cards(cards, self._hands[player], self._observation_hands[player])

    def _remove_from_hand(self, cards, player):
        self._remove_cards(cards, self._hands[player], self._observation_hands[player])

    def _add_to_taken(self, cards, player):
        if not hasattr(cards, '__iter__'): cards = [cards, ]
        for card in cards:
            self._rewards[player] += self.card_to_tricks(card)
        if len(self._taken[player]) < 27 and len(self._taken[player]) + len(cards) >= 27:
            self._rewards[player] += 3
        self._add_cards(cards, self._taken[player], self._observation_taken[player])

    def _deal_cards(self):
        self._add_to_hand(self._deck[self._deck_pointer:self._deck_pointer+self._hand_size], 0)
        self._deck_pointer += self._hand_size
        self._add_to_hand(self._deck[self._deck_pointer:self._deck_pointer+self._hand_size], 1)
        self._deck_pointer += self._hand_size

    def _start_game(self, initial_table_size):
        self._add_to_table(self._deck[self._deck_pointer:self._deck_pointer+initial_table_size])
        self._deck_pointer += initial_table_size
        self._deal_cards()

    def _update_game_state(self):
        if (not self._hands[0] and not self._hands[1]):
            if self._deck_pointer >= len(self._deck):
                self._is_terminal = True
                self._add_to_taken(self._table, self._last_to_take)
                self._remove_from_table(self.table)
            else:
                self._deal_cards()

    def play_card(self, card, take):
        assert self.is_valid_take(card, take), "Invalid take"
        assert not self._is_terminal, "Game is over"
        if not take:
            self._remove_from_hand(card, self._current_player)
            self._add_to_table(card)
        else:
            self._remove_from_hand(card, self._current_player)
            self._add_to_taken(card, self._current_player)
            self._remove_from_table(take)
            if len(self._table) == 0:
                self._rewards[self._current_player] += 1
            self._add_to_taken(take, self._current_player)
            self._last_to_take = self._current_player
        self._current_player = 1 - self._current_player
        self._move_counter += 1
        self._update_game_state()

    def print_state(self):
        print(f"Player 0: {self._hands[0]}, {self._observation_hands[0]}")
        print(f"Player 1: {self._hands[1]}, {self._observation_hands[1]}")
        print(f"Table: {self._table}, {self._observation_table}")
        print()
        print(f"Player 0 taken: {sorted(self._taken[0])}, {self._observation_taken[0]}")
        print(f"Player 1 taken: {sorted(self._taken[1])}, {self._observation_taken[1]}")