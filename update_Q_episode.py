import numpy as np

class update_Q_episode():

    def update_Qtable_on_episode_basis(Qtable_path,round_his_record_path,win_flag,gamma,learning_rate_for_episode_update):
        Qtable = np.load(Qtable_path)
        file = open(round_his_record_path,mode='r')
        round_his_as_string = file.read()
        round_his_as_list_of_tuple = convert_round_history_to_tuple(round_his_as_string)
        file.close()
        if win_flag:
            reward = 100
        else:
            reward = -100
        for i in range(len(round_his_as_list_of_tuple) - 1,0, -1):
            h = round_his_as_list_of_tuple[i-1]
            next_h = round_his_as_list_of_tuple[i]
            learning_target = reward + gamma * np.max(Qtable[next_h[0], :]) - Qtable[h]
            Qtable[h] = Qtable[h] + learning_rate_for_episode_update * learning_target
            reward = 0
        np.save(Qtable_path, Qtable)
        
def move_empty_string_from_list(List):
        count = 0
        for i in List:
            if(i==''):
                List.pop(count)
            count = count + 1
        return List


def convert_round_history_to_tuple(round_history_as_string):
    round_his = round_history_as_string.split('#')
    round_his = move_empty_string_from_list(round_his)
    round_his_as_list_of_tup = []
    for action_state in round_his:
        action_state = action_state.split('$')
        action_state = move_empty_string_from_list(action_state)
        count = 0
        for num in action_state:
            action_state[count] = int(num)
            count = count + 1
        action_state = tuple(action_state)
        round_his_as_list_of_tup.append(action_state)
    return round_his_as_list_of_tup
