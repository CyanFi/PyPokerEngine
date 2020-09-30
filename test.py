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
from my_players.DQNPlayer import DQNPlayer

num_episode = 10000
win = 0
ql_path = 'model/ql_z.npy'
model_path = 'model/DQN00.dump'
optimizer_path = 'model/DQN00.optim.dump'
count = 0
log_interval = 10
log = []
for i in range(0, num_episode):
    count = count + 1
    config = setup_config(max_round=15, initial_stack=100, small_blind_amount=5)
    config.register_player(name="p1", algorithm=RandomPlayer())
    config.register_player(name="p2",
                           algorithm=DQNPlayer(model_path=model_path, optimizer_path=optimizer_path, training=False))

    game_result = start_poker(config, verbose=0)
    if game_result['players'][1]['stack'] > game_result['players'][0]['stack']:
        # player 2 wins
        win += 1

    if count % log_interval == 0:
        print(count, ' episode ', win / count)
