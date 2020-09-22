# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer

num_episode = 500
win = 0
<<<<<<< Updated upstream
# path0 = 'model/QLearningPlayer_1.npy'
path0 ='model/ql.npy'
=======
<<<<<<< HEAD
<<<<<<< HEAD
path0 = 'model/de_ep_norm_20.npy'
#path1 = 'model/QLearningPlayer1.npy'
=======
# path0 = 'model/QLearningPlayer_1.npy'
path0 ='model/ql.npy'
>>>>>>> c6f893c6917339457f7522a3cc54ce89bd497f22
=======
# path0 = 'model/QLearningPlayer_1.npy'
path0 ='model/ql.npy'
>>>>>>> c6f893c6917339457f7522a3cc54ce89bd497f22
>>>>>>> Stashed changes
count = 0
log_interval = 5
log = []
for i in range(0, num_episode):
    count = i + 1
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=QLearningPlayer(path0, training=False,epsilon=0))
    game_result = start_poker(config, verbose=0)
    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        # if player 1 wins
        win += 1

    if count % log_interval == 0:
        print(count, ' episode ', win / count)
