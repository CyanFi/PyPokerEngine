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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyper-parameters
batch_size = 32
learning_rate = 1e-4
gamma = 0.9
exp_replay_size = 100000
epsilon = 0.1
learn_start = 0


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

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)


class DQNPlayer(QLearningPlayer):

    def __init__(self, path, training):
        """
        State: hole_card, community_card, self.stack, opponent_player.action
        """
        # training device: cpu / cuda
        self.device = device
        self.fold_ratio = self.raise_ratio = self.call_ratio = 1.0 / 3
        self.nb_player = self.player_id = None

        # hyper-parameter for q learning
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.experience_replay_size = exp_replay_size
        self.batch_size = batch_size
        self.learn_start = learn_start
        # training-required game attribute
        self.hand_strength = 0
        self.hole_card = None
        self.model_path = path
        self.update_count = 0
        self.history = []
        self.training = training
        # declare DQN model
        self.num_actions = 3
        # TODO, input shape
        self.num_feats = None
        self.declare_networks()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []
        self.rewards = []
        self.sigma_parameter_mag = []
        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        if self.training:
            self.model.train()
            self.target_model.train()
        else:
            self.model.eval()
            self.target_model.eval()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

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
                                      dtype=torch.uint8)
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
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.training:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start:
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
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or self.training:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        # TODO
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)


    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio


    def declare_action(self, valid_actions, hole_card, round_state):
        state = int(self.hand_strength * 20), round_state['big_blind_pos'], int(
            round_state['seats'][self.player_id]['stack'] / 10)

        # epsilon-greedy exploration
        if self.training and rand.random() < self.epsilon:
            choice = self.__choice_action(valid_actions)
        else:
            max_a = 0
            max_q = -100000
            for a in valid_actions:
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

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.history and self.training:
            # if player has declared some action before

            # define the reward and append the last action to history
            if winners[0]['uuid'] == self.uuid:
                # player win the game
                reward = winners[0]['stack'] - 100
                hand_strength = 20
            else:
                reward = 100 - winners[0]['stack']
                hand_strength = 0
            _h = hand_strength, round_state['big_blind_pos'], int(round_state['seats'][self.player_id]['stack'] / 10)
            self.history.append(_h + (None,))

            # reward all history actions
            for i in range(0, len(self.history) - 1):
                h = self.history[i]
                next_h = self.history[i + 1]
                learning_target = reward + self.gamma * np.max(self.Q[next_h[0], :]) - self.Q[h]
                self.Q[h] = self.Q[h] + self.learning_rate * learning_target
            # clear history
            self.history = []
            # save model
            np.save(self.model_path, self.Q)
