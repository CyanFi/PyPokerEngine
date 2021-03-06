# Train Script
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.AllCall import AllCallPlayer
from my_players.QLearningPlayer import QLearningPlayer
from my_players.HumanPlayer import ConsolePlayer
from my_players.DQNPlayer import DQNPlayer
from my_players.cardplayer import cardplayer
from my_players.A2CPlayer import A2CPlayer
from scipy.stats import t
import math
import os

num_episode = 50000
win = 0
sample_mean = 0
SXX = 0
sample_std = 0
path0 = os.getcwd() + '/model/a2c_1.dump'
path1 = os.getcwd() + '/model/a2c_1_optim.dump'
dqn_path0 = os.getcwd() + '/model/dqn.dump'
dqn_path1 = os.getcwd() + '/model/dqn_optim.dump'
count = 0
log_interval = 10
log = []
confidence_level = 0.05

config = setup_config(max_round=100, initial_stack=1500, small_blind_amount=5)
config.register_player(name="p1",
                       algorithm=cardplayer())
config.register_player(name="p2", algorithm=A2CPlayer(path0, path1, True))

for i in range(0, num_episode):
    count = count + 1
    game_result = start_poker(config, verbose=0)
    win = (game_result['players'][1]['stack'] -
           game_result['players'][0]['stack']) / 2 / 10
    # calculate bb / 100g
    last_mean = sample_mean
    sample_mean = sample_mean + (win - sample_mean) / count
    SXX = SXX + (win - sample_mean) * (win - last_mean)
    if count != 1:
        sample_std = math.sqrt(SXX / (count - 1))
    interval = t.interval(confidence_level, count - 1, sample_mean, sample_std)
    log.append((count, sample_mean))
    # if count == 1:
    #     _m = config.players_info[0]['algorithm'].declare_memory()
    #     print("Device:", config.players_info[0]['algorithm'].device)
    #     print("Device:", config.players_info[0]['algorithm'].device)
    # else:
    #     config.players_info[0]['algorithm'].memory = _m

    if count % log_interval == 0:
        print(count, ' episode, 百手盈利', sample_mean, u"\u00B1",
              (interval[1] - interval[0]) / 2)
        config.players_info[1]['algorithm'].save_model()
        # config.players_info[0]['algorithm'].save_model()

# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_axis = [item[0] for item in log]
ax.plot(x_axis, [item[1] for item in log], label='A2C agent')
ax.set_xlabel('episode')
ax.set_ylabel('BB / 100g')
ax.legend()
ax.set_title('NLH Poker Game result (A2C agent vs Card)')
plt.show()

# print(log_loss)
# fig, ax = plt.subplots()
# x_axis = [i for i in range(0,len(log_loss))]
# ax.plot(x_axis, [i for i in log_loss], label='A2C agent')
# ax.set_xlabel('episode')
# ax.set_ylabel('loss')
# ax.legend()
# ax.set_title('NLH Poker Game result (A2C agent vs card)')
# plt.show()
