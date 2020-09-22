\# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer

num_episode = 300
win = 0
path0 = 'model/de_ep_norm_20.npy'
#path1 = 'model/QLearningPlayer1.npy'
count = 0
log_interval = 20
log = []
for i in range(0, num_episode):
    count = count+1
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=QLearningPlayer(path0, training=False,epsilon=0))
    game_result = start_poker(config, verbose=0)
    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        # player 2 wins
        win += 1
        #print('episode {} winner: player {}'.format(i + 1, 2))
    #else:
        #print('episode {} winner: player {}'.format(i + 1, 1))
    if count % log_interval == 0:
        log.append([i + 1, win / count])
        print(count,' episode ',win / count)
