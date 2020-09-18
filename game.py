# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    game.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer

path='model/ql.npy'
for i in range(0,100):
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=20)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=QLearningPlayer(path))
    game_result = start_poker(config, verbose=3)
