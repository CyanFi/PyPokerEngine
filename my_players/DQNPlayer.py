# GOOD GOOD STUDY, DAY DAY UP!
#    @Time:    9/18/2020 5:03 PM
#    @Author:  qingy
#    @Email:   qingyuge006@gmail.com
#    @File:    QLearningPlayer.py
#    @Project: nlh-poker
#    @Description: DQN Agent adapted from https://github.com/cuhkrlcourse/DeepRL-Tutorials/blob/master/01.DQN.ipynb
from .QLearningPlayer import QLearningPlayer
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# hyper-parameters
batch_size = 32
learning_rate = 1e-4
gamma = 0.99
exp_replay_size = 10000
epsilon = 0.1
learn_start = 100
target_net_update_freq = 500


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Deep Q Network
class DQN(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_shape[0], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNPlayer(QLearningPlayer):

    def __init__(self, model_path, optimizer_path, training):
        """
        State: hole_card, community_card, self.stack, opponent_player.action
        """

        # training device: cpu > cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fold_ratio = self.raise_ratio = self.call_ratio = 1.0 / 3
        self.nb_player = self.player_id = None

        # hyper-parameter for Deep Q Learning
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.experience_replay_size = exp_replay_size
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_net_update_freq = target_net_update_freq
        # training-required game attribute
        self.stack = 100
        self.hole_card = None
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.update_count = 0
        self.history = []
        self.training = training
        # declare DQN model
        self.num_actions = 3
        self.num_feats = (10,)
        self.declare_networks()
        try:
            self.model.load_state_dict(torch.load(self.model_path))
        except:
            pass
        self.target_model.load_state_dict(self.model.state_dict())
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        try:
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
        except:
            pass
        self.losses = []
        self.sigma_parameter_mag = []
        if self.training:
            self.model.train()
            self.target_model.train()
        else:
            self.model.eval()
            self.target_model.eval()

        self.update_count = 0
        self.episode = 0

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
        return self.memory

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'sigma' in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        # estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            # To prevent tracking history of gradient
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, episode=0):
        if not self.training:
            return None

        self.append_to_replay(s, a, r, s_)

        if episode < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        # self.save_loss(loss.item())
        # self.save_sigma_param_magnitudes()

    def update_target_model(self):
        """
        to use in fix-target
        """
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    @staticmethod
    def huber(x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    @staticmethod
    def bce_loss(x, y):
        """
        :return: binary entropy loss between x and y
        """
        _x = 1 / (1 + torch.exp(-x))
        _y = 1 / (1 + torch.exp(-y))
        return -(_y * torch.log(_x) + (1 - _y) * torch.log(1 - _x))

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

    def eps_greedy_policy(self, s, opponent, valid_actions, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or not self.training:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                action_list = self.model(X).cpu().numpy().reshape(-1)
                if opponent['state'] == 'allin' or valid_actions[2]['amount']['max'] == -1:
                    action_list = np.delete(action_list, 2)
                if valid_actions[1]['amount'] == 0:
                    # if one can check, do not fold
                    action_list[0] = action_list[1] - 1
                return np.argmax(action_list)
            else:
                action_list = np.array([0, 1, 2])
                if opponent['state'] == 'allin' or valid_actions[2]['amount']['max'] == -1:
                    action_list = np.delete(action_list, 2)
                if valid_actions[1]['amount'] == 0:
                    # if one can check, do not fold
                    action_list = np.delete(action_list, 0)
                return np.random.choice(action_list, 1).item()

    @staticmethod
    def process_state(s):
        new_s = list(s)
        for i in range(0, 7):
            new_s[i] = new_s[i] / 26.0 - 1
        for i in range(7, 10):
            new_s[i] = (new_s[i] - 100) / 100.0
        return tuple(new_s)

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        state: hole_card, community_card, self.stack
        """
        # preprocess variable in states
        hole_card_1 = self.card_to_int(hole_card[0])
        hole_card_2 = self.card_to_int(hole_card[1])
        self.hole_card = (hole_card_1, hole_card_2)
        community_card = self.community_card_to_tuple(round_state['community_card'])

        state = self.hole_card + community_card + (int(round_state['seats'][self.player_id]['stack']),) + (
            valid_actions[1]['amount'], valid_actions[2]['amount']['min'])
        state = self.process_state(state)
        action = self.eps_greedy_policy(state, round_state['seats'][(self.player_id + 1) % 2], valid_actions,
                                        self.epsilon)
        action = valid_actions[action]['action']
        if action == "raise":
            # To simplify the problem, raise only at minimum
            amount = valid_actions[2]["amount"]["min"]
        elif action == "call":
            amount = valid_actions[1]["amount"]
        else:
            amount = 0
        # record the action
        self.history.append(state + (self.action_to_int(action),))
        return action, amount

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
        pass

    @staticmethod
    def round_int_to_string(round_int):
        m = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return m[round_int]

    def receive_round_result_message(self, winners, hand_info, round_state):
        if len(self.history) >= 1 and self.training:
            # if player has declared some action before

            # define the reward and append the last action to history
            if winners[0]['uuid'] == self.uuid:
                # player win the game
                reward = winners[0]['stack'] - self.stack
                self.stack = winners[0]['stack']
            else:
                new_stack = 200 - winners[0]['stack']
                reward = new_stack - self.stack
                self.stack = new_stack
            # average reward
            reward /= len(self.history)

            action_num = len(round_state['action_histories'])
            if round_state['action_histories'][self.round_int_to_string(action_num - 1)] == []:
                action_num -= 1
            if action_num == 1:
                community_card = self.community_card_to_tuple([])
            elif action_num == 2:
                community_card = self.community_card_to_tuple(round_state['community_card'][:3])
            elif action_num == 3:
                community_card = self.community_card_to_tuple(round_state['community_card'][:4])
            elif action_num == 4:
                community_card = self.community_card_to_tuple(round_state['community_card'][:5])
            last_state = (self.hole_card[0], self.hole_card[1]) + community_card + (round_state['seats'][self.player_id]['stack'],) + (-1, -1)
            last_state = self.process_state(last_state)
            # append the last state to history
            self.history.append(last_state + (None,))

            # update using reward
            for i in range(0, len(self.history) - 1):
                h = self.history[i]
                next_h = self.history[i + 1]
                self.update(h[:-1], h[-1], reward, next_h[:-1], self.episode)
            # clear history
            self.history = []

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
