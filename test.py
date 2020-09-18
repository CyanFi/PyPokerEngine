# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer

num_episode = 100
win = 0
path = 'model/ql.npy'
for i in range(0, num_episode):
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=QLearningPlayer(path, training=False))
    game_result = start_poker(config, verbose=0)
    if game_result['players'][1]['stack'] != 0:
        # player 1 wins
        win += 1
print("winning {} out of {} episodes".format(win, num_episode))
