'''
Date: 2020-11-07 13:53:29
Editor: qingy
LastEditTime: 2020-11-07 13:53:54
FilePath: /a2c/my_players/AllCall.py
Description: This is file generated by AC automation machine. Qingy knows nothing.
'''
from pypokerengine.players import BasePokerPlayer


class AllCallPlayer(BasePokerPlayer):

    def __init__(self):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        return valid_actions[1]['action'], valid_actions[1]['amount']

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