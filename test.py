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
# from my_players.HonestPlayer import HonestPlayer
from scipy.stats import t
import math
num_episode = 500
win = 0
path0 = 'model/testing.npy'
count = 0
log_interval = 10
log = []
confidence_level = 0.95

for i in range(0, num_episode):
    count = count + 1
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=QLearningPlayer(path0, training=False))
    config.players_info[1]['algorithm'].oponent = config.players_info[0]['algorithm']
    config.players_info[0]['algorithm'].oponent = config.players_info[1]['algorithm']
    game_result = start_poker(config, verbose=0)
    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        # player 2 wins
        win += 1

    last_mean = sample_mean
    sample_mean = sample_mean + (win - sample_mean) / count
    SXX = SXX + (win - sample_mean) * (win - last_mean)
    if count != 1:
        sample_std = math.sqrt(SXX / (count - 1))
    interval = t.interval(confidence_level, count - 1, sample_mean, sample_std)

    if count % log_interval == 0:
        print(count, ' episode, 百手盈利', sample_mean, u"\u00B1", (interval[1] - interval[0]) / 2)
