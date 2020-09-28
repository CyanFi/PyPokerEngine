# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer
from my_players.DQNPlayer import DQNPlayer
import math

num_episode = 100000
log_interval = 100

# DQN model hyper-parameters: epsilon variables
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10000
epsilon_decrease = lambda episode_idx: 0.00

print('Training episode: {}.\nLog every {} episode.\n'.format(num_episode, log_interval))
# model path
model_path = 'model/DQN5.dump'
optimizer_path = 'model/DQN5_optim.dump'

win = 0
log = []

for i in range(0, num_episode):
    count = i + 1
    config = setup_config(max_round=15, initial_stack=100, small_blind_amount=5)
    # The first player is random player
    config.register_player(name="p1", algorithm=RandomPlayer())
    # THe second player is training
    config.register_player(name="p2",
                           algorithm=DQNPlayer(model_path=model_path, optimizer_path=optimizer_path, training=True))

    # update epsilon
    config.players_info[1]['algorithm'].epsilon = epsilon_decrease(count)
    # update episode num
    config.players_info[1]['algorithm'].episode = count
    if count == 1:
        _m = config.players_info[1]['algorithm'].declare_memory()
        print("Device:", config.players_info[1]['algorithm'].device)
    else:
        config.players_info[1]['algorithm'].memory = _m
    game_result = start_poker(config, verbose=0)

    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        # if player 1 wins
        win += 1
    if count % log_interval == 0:
        log.append([i + 1, win / count])
        print(count, ' episode ', win / count)
        # save model
        config.players_info[1]['algorithm'].save_model()

# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_axis = [item[0] for item in log]
ax.plot(x_axis, [item[1] for item in log], label='DQN agent')
ax.set_xlabel('episode')
ax.set_ylabel('winning rate per episode')
ax.legend()
ax.set_title('NLH Poker Game result (DQN agent vs random)')
plt.show()
