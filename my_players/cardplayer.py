from pypokerengine.players import BasePokerPlayer
import random as rand
import numpy as np
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


class cardplayer(BasePokerPlayer):

    def __init__(self):
        self.hand_strength = 0
        self.num_simulation = 100

    def load_model(self):
        pass

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
                                                         nb_player=2,
                                                         hole_card=gen_cards(hole_card),
                                                         community_card=gen_cards(round_state['community_card']))
#         state = int(self.hand_strength * 10), round_state['big_blind_pos'], int(
#             round_state['seats'][self.player_id]['stack'] / 10)
        final_valid_actions = valid_actions
        if final_valid_actions[2]['amount']['max'] == -1:
            #final_valid_actions.pop(2)
            if self.hand_strength<0.1:
                action = final_valid_actions[0]["action"]
                amount = final_valid_actions[0]["amount"]
            else:
                action = final_valid_actions[1]["action"]
                amount = final_valid_actions[1]["amount"]
        else:
            if self.hand_strength<0.1:
                action = final_valid_actions[0]["action"]
                amount = final_valid_actions[0]["amount"]
            elif self.hand_strength<0.9:
                action = final_valid_actions[1]["action"]
                amount = final_valid_actions[1]["amount"]
            else:
                action = final_valid_actions[2]["action"]
                amount = final_valid_actions[2]["amount"]['max']
        
        return action, amount

    def __choice_action(self, valid_actions):
        pass

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass