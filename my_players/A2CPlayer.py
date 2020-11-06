# based on https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
from pypokerengine.players import BasePokerPlayer
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# use cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Training device:", device)


class ActorCritic(nn.Module):
    # neural network here
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


class A2CPlayer(BasePokerPlayer):

    def __init__(self, model_path, optimizer_path, training):
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.training = training
        self.stack = 1500
        self.hole_card = None

        self.lr = 1e-4
        self.gamma = 0.95
        self.num_inputs = 8  # 2 hold card, 5 community card, self.stack
        self.num_outputs = 4  # fold, call, raise min, raise max
        self.model = ActorCritic(self.num_inputs, self.num_outputs, 128)
        self.optimizer = optim.Adam(self.model.parameters())

        # refresh every cycle
        self.history = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropy = 0
        # load model
        try:
            self.model.load_state_dict(torch.load(self.model_path))
        except:
            pass
        try:
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
        except:
            pass

    def discrete_action(self, state, valid_actions):
        dist, value = self.model(state)
        action_raw = dist.sample()
        if action_raw < 3:
            action = valid_actions[action_raw]['action']
            if action == "raise":
                # To simplify the problem, raise only at minimum
                amount = valid_actions[2]["amount"]["min"]
            elif action == "call":
                amount = valid_actions[1]["amount"]
            else:
                amount = 0

        else:
            action = "raise"
            amount = valid_actions[2]['amount']['max']
        return action_raw, action, amount, dist, value

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        state: hole_card, community_card, self.stack
        """
        # preprocess states
        # self.episode += 1
        hole_card_1 = self.card_to_int(hole_card[0])
        hole_card_2 = self.card_to_int(hole_card[1])
        self.hole_card = (hole_card_1, hole_card_2)
        community_card = self.community_card_to_tuple(
            round_state['community_card'])

        state = self.hole_card + community_card + \
            (int(round_state['seats'][self.player_id]['stack']/10),)

        state = self.process_state(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_int, action, amount, dist, value = self.discrete_action(
            state, valid_actions)

        # record the trajactory
        done = False
        log_prob = dist.log_prob(action_int)
        self.entropy += dist.entropy().mean()
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(device))
        # self.history.append(state + (action_int, int(amount/10)))

        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        # end of a cycle
        if len(self.values) >= 1 and self.training:
            # if player has declared some action before

            # calulate reward
            if winners[0]['uuid'] == self.uuid:
                # player win the game
                reward = winners[0]['stack'] - self.stack
                self.stack = winners[0]['stack']
            else:
                new_stack = 3000 - winners[0]['stack']
                reward = new_stack - self.stack
                self.stack = new_stack

            reward /= 10
            self.rewards = [reward]*len(self.values)
            self.masks[-1] = True

            # preprocess the last state
            action_num = len(round_state['action_histories'])
            if round_state['action_histories'][self.round_int_to_string(action_num - 1)] == []:
                action_num -= 1
            if action_num == 1:
                community_card = self.community_card_to_tuple([])
            elif action_num == 2:
                community_card = self.community_card_to_tuple(
                    round_state['community_card'][:3])
            elif action_num == 3:
                community_card = self.community_card_to_tuple(
                    round_state['community_card'][:4])
            elif action_num == 4:
                community_card = self.community_card_to_tuple(
                    round_state['community_card'][:5])
            
            last_state = (self.hole_card[0], self.hole_card[1]) + community_card + (
                int(round_state['seats'][self.player_id]['stack']/10),)
            last_state = self.process_state(last_state)
            last_state = torch.FloatTensor(last_state).unsqueeze(0).to(device)
            # format: state, action_type, action_amount, dist, value, mask
            # self.masks.append(True)
            _, last_value = self.model(last_state)
            # self.history.append(last_state + (None, None))

            # calculate loss
            self.returns = self.compute_returns(last_value)
            self.log_probs = torch.cat(self.log_probs)
            self.returns = torch.cat(self.returns).detach()
            self.values = torch.cat(self.values)
            advantage = self.returns-self.values
            actor_loss = -(self.log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss+0.5*critic_loss - 0.01*self.entropy
            print("loss:",loss)

            # back prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # reset value
            self.history = []
            self.log_probs = []
            self.values = []
            self.rewards = []
            self.masks = []
            self.entropy = 0
            # self.save_model()

    def compute_returns(self, final_value):
        R = final_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)

    # ultility functions

    @staticmethod
    def process_state(s):
        new_s = list(s)
        for i in range(0, 7):
            new_s[i] = new_s[i] / 26.0 - 1
        for i in range(7, 8):
            new_s[i] = (new_s[i] - 150) / 150.0
        return tuple(new_s)

    @staticmethod
    def card_to_int(card):
        """convert card to int, card[0]:花色, card[1]:rank"""
        suit_map = {'H': 0, 'S': 1, 'D': 2, 'C': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11,
                    'A': 12}
        return suit_map[card[0]] * 13 + rank_map[card[1]]

    def community_card_to_tuple(self, community_card):
        """
        :param community_card: round_state['community_card']
        :return: tuple of int (0..52)
        """
        new_community_card = []
        for i in range(0, len(community_card)):
            new_community_card.append(self.card_to_int(community_card[i]))
        for i in range(0, 5 - len(community_card)):
            # if community card num <5, append 52 to fill out the rest
            new_community_card.append(52)
        return tuple(new_community_card)

    @staticmethod
    def round_int_to_string(round_int):
        m = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return m[round_int]

    def receive_game_start_message(self, game_info):
        for i in range(0, len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i
                break

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass