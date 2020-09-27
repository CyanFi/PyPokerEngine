from pypokerengine.players import BasePokerPlayer
import random as rand

class RandomPlayer(BasePokerPlayer):

  def __init__(self):
    self.fold_ratio = self.call_ratio = raise_ratio = 1.0/3
    self.oponent = None
    self.cur_stack = 100

  def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
    ratio = [fold_ratio, call_ratio, raise_ratio]
    scaled_ratio = [ 1.0 * num / sum(ratio) for num in ratio]
    self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

  def declare_action(self, valid_actions, hole_card, round_state):
    final_valid_actions=valid_actions
    if final_valid_actions[2]['amount']['max'] == -1 or self.oponent.cur_stack == 0:
        final_valid_actions.pop(2)
    choice = self.__choice_action(valid_actions)
    action = choice["action"]
    amount = choice["amount"]
    if action == "raise":
      amount = rand.randrange(amount["min"], max(amount["min"], amount["max"]) + 1)
    return action, amount

  def __choice_action(self, valid_actions):
    r = rand.random()
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
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, new_action, round_state):
    self.cur_stack = round_state['seats'][0]['stack']

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass