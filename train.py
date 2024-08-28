from tablic import Tablic
from dqn.dqnAgent import DQNAgent
from players.qPlayer import QPlayer
from plot import Tracker
import numpy as np
import time
import random

# Number of episodes
EPISODES = 50000
SAVE_FREQ = 5000

# Update frequency
UPDATE_FREQ = 5
# Switch nets frequency
SWITCH_FREQ = 50
# Plot frequency
PLOT_FREQ=1000

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / EPISODES

# Gamma
GAMMA = 0

# Device
DEVICE="cpu"

# Checkpoint path
chckPointPath=None
if __name__ == '__main__':
    random.seed(0)
    tracker=Tracker(100)
    for ALPHA in [0.001]:
        for GAMMA in [0.95,1]:
            print(f"Training with gamma {GAMMA} started and alpha {ALPHA}.")
            agent=DQNAgent(GAMMA,device=DEVICE,alpha=ALPHA,multiStep=False,prioritized=True)
            player = QPlayer(agent)
            epsilon = EPSILON_START
            startEpisode=1
            start = time.time()

            if chckPointPath:
                trainDict=agent.loadCheckpoint(chckPointPath)
                startEpisode=trainDict['episode']
                epsilon=trainDict['epsilon']
                EPISODES=trainDict['EPISODES']
                SAVE_FREQ=trainDict['SAVE_FREQ']
                UPDATE_FREQ=trainDict['UPDATE_FREQ']
                SWITCH_FREQ=trainDict['SWITCH_FREQ']
                EPSILON_END=trainDict['EPSILON_END']
                EPSILON_DECAY=trainDict['EPSILON_DECAY']
                elapsedTime=trainDict['elapsedTime']

                start=start-elapsedTime

                random.setstate(trainDict['rngState'])

            for episode in range(startEpisode, EPISODES+1):
                deck= Tablic.getShuffledDeck()
                game = Tablic(deck=deck)
                game_actions = [[],[]]
                game_rewards = [[],[]]
                game_valid_actions = [[],[]]

                if (episode > SWITCH_FREQ and episode % SWITCH_FREQ == 0):
                    agent.updateTarget()

                while not game.isTerminal:
                    current_player = game.currentPlayer

                    all_state_actions = game.allValidStateActions(game.table,game.hands[game.currentPlayer],game.getGameStateRepresentation())
                    game_valid_actions[current_player].append(all_state_actions)
                    game_rewards[current_player].append(game.rewards[current_player])

                    if random.random() < epsilon:
                        played_card, played_take = player.playerRandomPlay(game)
                    else:
                        played_card, played_take = player.playerPolicyPlay(game)

                    state_action = np.concatenate((game.getActionRepresentation(played_card,played_take),game.getGameStateRepresentation()))
                    game.playCard(played_card, list(played_take))
                    game_actions[current_player].append(state_action)

                for current_player in range(2):
                    game_rewards[current_player].append(game.rewards[current_player])
                    game_valid_actions[current_player].append([])
                    for i in range(len(game_actions[current_player])):
                        action = game_actions[current_player][i]
                        reward = game_rewards[current_player][i+1] - game_rewards[current_player][i]
                        valid_actions = game_valid_actions[current_player][i+1]
                        player.agent.remember(action, reward, valid_actions)
                
                if episode % UPDATE_FREQ == 0:
                    print(f"Time needed for {episode} episodes with {GAMMA} and {ALPHA}: {time.time()-start}")
                    agent.backward()

                if episode>=PLOT_FREQ and (episode % PLOT_FREQ == 0):
                    tracker.evaluate(player,alpha=ALPHA,gamma=GAMMA,episode=episode)

                if episode % SAVE_FREQ == 0:
                    print(f"Episode {episode} saved.")
                    agent.saveModelsParams(f"models/minimalModelParams/a{int(ALPHA*10000)}g{int(GAMMA*100)}e{episode}")
                    agent.saveCheckpoint({
                        'episode': episode+1,
                        'epsilon' : max(epsilon - EPSILON_DECAY, EPSILON_END),
                        'EPISODES' : EPISODES,
                        'SAVE_FREQ' : SAVE_FREQ,
                        'UPDATE_FREQ' : UPDATE_FREQ,
                        'SWITCH_FREQ' : SWITCH_FREQ,
                        'EPSILON_END' : EPSILON_END,
                        'EPSILON_DECAY' : EPSILON_DECAY,
                        'rngState': random.getstate(),
                        'elapsedTime' : time.time()-start,
                    },f"models/minimalModelParams/checkpoint{int(ALPHA*10000)}{int(GAMMA*100)}e{episode}")
                epsilon = max(epsilon - EPSILON_DECAY, EPSILON_END)

            end = time.time()
            print(f"Training with GAMMA={GAMMA} lasted {end-start} seconds.")
