# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    test.py
#    @Project: pypokerengine
from random import random
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.AllCall import AllCallPlayer
from my_players.QLearningPlayer import QLearningPlayer
from my_players.HumanPlayer import ConsolePlayer
from my_players.DQNPlayer import DQNPlayer
from my_players.cardplayer import cardplayer
from my_players.A2CPlayer import A2CPlayer
from my_players.HonestPlayer import HonestPlayer
from scipy.stats import t
import math

num_episode = 10000
win = 0
sample_mean = 0
SXX = 0
sample_std = 0
model_path = 'model/a2c_1.dump'
optimizer_path = 'model/a2c_1_optim.dump'
count = 0
log_interval = 10
log = []
confidence_level = 0.05
config = setup_config(max_round=100, initial_stack=1500, small_blind_amount=5)
config.register_player(name="p1", algorithm=HonestPlayer())
config.register_player(name="p2",
                        algorithm=A2CPlayer(model_path,optimizer_path,False))
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

    if count % log_interval == 0:
        print('honest vs a2c',count, ' episode, 百手盈利', sample_mean, u"\u00B1", (interval[1] - interval[0]) / 2)