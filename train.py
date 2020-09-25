# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 4:52 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    train.py.py
#    @Project: pypokerengine
from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.QLearningPlayer import QLearningPlayer
import numpy as np
from update_Q_episode import update_Q_episode


num_episode = 5000
log_interval = 20
update_interval = 50
print('Training episode: {}.\nLog every {} episode.\n'.format(num_episode, log_interval))
path = 'model/ql2.npy'
# path1 = 'model/QLearningPlayer1.npy'
win = 0
log = []
round_his_record = 'round_his_record.txt'
GAMMA = 0.9
for i in range(0, num_episode):
    fp = open(round_his_record,'w+')
    fp.truncate()  #clear file
    fp.close()
    count = i + 1
    config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)

    # The first player is random player
    config.register_player(name="p1", algorithm=RandomPlayer())
    # THe second player is training
    config.register_player(name="p2", algorithm=QLearningPlayer(path, training=True,round_his_record=round_his_record))
    # update epsilon
    config.players_info[1]['algorithm'].epsilon = 1 / (count/10+1)
    game_result = start_poker(config, verbose=0)

    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        # if player 1 wins
        win += 1
        win_flag = True
    else:
        win_flag = False
    update_Q_episode.update_Qtable_on_episode_basis(Qtable_path=path,round_his_record_path=round_his_record,win_flag=win_flag,gamma=GAMMA,learning_rate_for_episode_update=0.02)
    if count % log_interval == 0:
        log.append([i + 1, win / count])
        print(count, ' episode ', win / count)


    
    
# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_axis = [item[0] for item in log]
ax.plot(x_axis, [item[1] for item in log], label='Q-learning agent')
ax.set_xlabel('episode')
ax.set_ylabel('winning rate per episode')
ax.legend()
ax.set_title('NLH Poker Game result (Q Learning agent vs random)')
plt.show()
