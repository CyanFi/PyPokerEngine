# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer
from my_players.HumanPlayer import ConsolePlayer
from my_players.HonestPlayer import HonestPlayer
from my_players.AllCallPlayer import AllCallPlayer
from my_players.DQNPlayer import DQNPlayer
from my_players.cardplayer import cardplayer
from scipy.stats import t
import math

num_episode = 50000
win = 0
sample_mean = 0
SXX = 0
sample_std = 0
path0 = 'model/DQN_card_1.npy'
path1 = 'model/DQN4_optim_1.dump'
count = 0
log_interval = 10
log = []
log_loss = []
confidence_level = 0.95
memory = None
config = setup_config(max_round=100, initial_stack=1500, small_blind_amount=5)
config.register_player(name="p1", algorithm=cardplayer())
config.register_player(name="p2", algorithm=DQNPlayer(model_path=path0,optimizer_path=path1,training=True))
config.players_info[1]['algorithm'].oponent = config.players_info[0]['algorithm']
for i in range(0, num_episode):
    count = count + 1
    game_result = start_poker(config, verbose=0)
    win = (game_result['players'][1]['stack'] - game_result['players'][0]['stack']) / 2 / 10
    last_mean = sample_mean
    sample_mean = sample_mean + (win - sample_mean) / count
    SXX = SXX + (win - sample_mean) * (win - last_mean)
    if count != 1:
        sample_std = math.sqrt(SXX / (count - 1))
    interval = t.interval(confidence_level, count - 1, sample_mean, sample_std)
    log.append((count,sample_mean))
    log_loss.append(config.players_info[1]['algorithm'].loss)
    if count % log_interval == 0:
        print(count, ' episode, 百手盈利', sample_mean, u"\u00B1", (interval[1] - interval[0]) / 2,"loss: ",config.players_info[1]['algorithm'].loss)

# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_axis = [item[0] for item in log]
ax.plot(x_axis, [item[1] for item in log], label='Deep Q-learning agent')
ax.set_xlabel('episode')
ax.set_ylabel('winning rate per episode')
ax.legend()
ax.set_title('NLH Poker Game result (Deep Q Learning agent vs card)')
plt.show()

print(log_loss)
fig, ax = plt.subplots()
x_axis = [i for i in range(0,len(log_loss))]
ax.plot(x_axis, [i for i in log_loss], label='Deep Q-learning agent')
ax.set_xlabel('episode')
ax.set_ylabel('loss')
ax.legend()
ax.set_title('NLH Poker Game result (Deep Q Learning agent vs card)')
plt.show()
