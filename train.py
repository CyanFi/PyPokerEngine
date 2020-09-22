# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer

num_episode = 3000
log_interval = 20
print('Training episode: {}.\nLog every {} episode.\n'.format(num_episode, log_interval))
path0 = 'model/de_ep_norm_20.npy'
#path1 = 'model/QLearningPlayer1.npy'
win = 0
log = []
flag = 0
count = 0
for i in range(0,num_episode):
    count = count + 1
    ep = 1 / (i+1)
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2", algorithm=QLearningPlayer(path0, training=True,epsilon=ep))
    game_result = start_poker(config, verbose=0)
    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        win += 1
    if count % log_interval == 0:
        log.append([i + 1, win / count])
        print(count,' episode ',win / count)


        
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_axis = [item[0] for item in log]
ax.plot(x_axis, [item[1] for item in log], label='Q-learning agent')
ax.set_xlabel('episode')
ax.set_ylabel('winning rate per episode')
ax.legend()
ax.set_title('NLH Poker Game result (Q Learning agent vs random)')
plt.show()