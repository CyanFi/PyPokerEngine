# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 5:03 PM
#    @Author:  Qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    QLearningPlayer.py
#    @Project: pypokerengine
from pypokerengine.players import BasePokerPlayer
import random as rand
import numpy as np
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


class QLearningPlayer(BasePokerPlayer):

    def __init__(self, path, training):
        """
        Q: hand_strength, big_blind_pos, self.stack, action
        """
        self.fold_ratio = self.raise_ratio = self.call_ratio = 1.0 / 3
        self.nb_player = self.player_id = None
        try:
            # load model:
            self.Q = np.load(path)
        except:
            # initialize Q table
            self.Q = np.zeros((11, 2, 21, 3))
            pass
        # hyper-parameter for q learning
        self.epsilon = 0.1
        self.gamma = 0.9
        self.learning_rate = 0.01
        self.num_simulation = 100
        # training-required game attribute
        self.hand_strength = 0
        self.stack = 100
        self.hole_card = None
        self.model_path = path
        self.history = []
        self.training = training
        self.oponent = None
        self.cur_stack = 100

    def load_model(self):
        self.Q = np.load(self.model_path)

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    @staticmethod
    def action_to_int(action):
        """
        convert type action to int
        """
        if action == 'fold':
            return 2
        if action == 'call':
            return 1
        if action == 'raise':
            return 0

    def declare_action(self, valid_actions, hole_card, round_state):
        self.hand_strength = estimate_hole_card_win_rate(nb_simulation=self.num_simulation,
                                                         nb_player=self.nb_player,
                                                         hole_card=gen_cards(hole_card),
                                                         community_card=gen_cards(round_state['community_card']))
        state = int(self.hand_strength * 10), round_state['big_blind_pos'], int(
            round_state['seats'][self.player_id]['stack'] / 10)
        final_valid_actions=valid_actions
        if final_valid_actions[2]['amount']['max'] == -1 or self.oponent.cur_stack == 0:
            final_valid_actions.pop(2)
        # epsilon-greedy exploration
        if self.training and rand.random() < self.epsilon and self.training:
            choice = self.__choice_action(final_valid_actions)
        else:
            max_a = 0
            max_q = -100000
            for a in final_valid_actions:
                tmp_a = self.action_to_int(a['action'])
                i = state + (tmp_a,)
                if self.Q[i] > max_q:
                    max_a = a
                    max_q = self.Q[i]
            choice = max_a

        action = choice["action"]
        amount = choice["amount"]
        if action == "raise":
            # To simplify the problem, raise only at minimum
            amount = amount["min"]
        # record the action
        self.history.append(state + (self.action_to_int(action),))
        
        return action, amount

    def __choice_action(self, valid_actions):
        if len(valid_actions)==3:
            r = rand.random()
            if r <= self.fold_ratio:
                return valid_actions[0]
            elif r <= self.call_ratio:
                return valid_actions[1]
            else:
                return valid_actions[2]
        else:
            if rand.random()<0.5:
                return  valid_actions[0]
            else:
                return valid_actions[1]

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        for i in range(0, len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i
                break

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        self.cur_stack = round_state['seats'][self.player_id]['stack']

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.history and self.training:
            # if player has declared some action before

            # define the reward and append the last action to history
            if winners[0]['uuid'] == self.uuid:
                # player win the game
                reward = winners[0]['stack'] - self.stack
                self.stack = winners[0]['stack']
                hand_strength = 10
            else:
                new_stack = 200 - winners[0]['stack']
                reward = new_stack - self.stack
                self.stack = new_stack
                hand_strength = 0
            _h = hand_strength, round_state['big_blind_pos'], int(round_state['seats'][self.player_id]['stack'] / 10)
            self.history.append(_h + (None,))

            # reward all history actions
            geometric_array = np.logspace(1, len(self.history) - 1, len(self.history) - 1, base=0.5)
            geometric_array[len(self.history) - 1] = geometric_array[len(self.history) - 2]
            reward_array = reward * geometric_array
            # reward /= len(self.history)
            for i in range(0, len(self.history) - 1):
                h = self.history[i]
                next_h = self.history[i + 1]
                learning_target = reward_array[i] + self.gamma * np.max(self.Q[next_h[0], :]) - self.Q[h]
                self.Q[h] = self.Q[h] + self.learning_rate * learning_target
            # clear history
            self.history = []
            # save model
            np.save(self.model_path, self.Q)
